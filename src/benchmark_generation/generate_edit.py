import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is not set. "
        "Please set it as an environment variable before running."
    )

if not OPENAI_BASE_URL:
    raise ValueError(
        "OPENAI_BASE_URL is not set. "
        "Please set it as an environment variable before running."
    )

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    http_client=cast(Any, httpx.Client()),
)


attribute_counter = defaultdict(int)
category_counter = defaultdict(int)


def call_openai(
    prompt,
    temperature=0.8,
    max_output_tokens=2048,
    system_prompt="You are a JSON generator.",
):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_output_tokens,
    )
    content = response.choices[0].message.content
    content = content.strip().replace("```json", "").replace("```", "")
    return content


NUM_STORIES = 20
SAVE_PATH = "image_editing_benchmark.json"

TEMPERATURE = 0.9


EDIT_TYPES = [
    "local_edit",
    "global_edit",
    "text_edit",
    "style_transfer",
    "character_edit",
]
EDIT_WEIGHTS = [0.25, 0.25, 0.2, 0.15, 0.15]
SUBJECT_CATEGORY_POOL = [
    "human",
    "animal",
    "robot",
    "plant",
    "alien",
    "monster",
    "cartoon character",
    "mythical creature",
    "living object",
]
ENVIRONMENT_TYPE_POOL = [
    "natural outdoor",
    "pure nature",
    "enclosed indoor",
    "urban open space",
    "domestic interior",
    "institutional interior",
    "fantasy setting",
]


def safe_json_load(content):
    try:
        return json.loads(content)
    except:
        return None


def generate_base_prompt(previous_scenes=None):
    subject_category = random.choice(SUBJECT_CATEGORY_POOL)
    environment_type = random.choice(ENVIRONMENT_TYPE_POOL)
    previous_scenes = [
        str(scene).strip()
        for scene in (previous_scenes or [])
        if isinstance(scene, str) and scene.strip()
    ][-10:]
    previous_scenes_text = (
        "\nPreviously generated base scenes (do not reuse their subject, setting, "
        "main props, or composition):\n- " + "\n- ".join(previous_scenes)
        if previous_scenes
        else ""
    )
    prompt = f"""
    Generate a detailed image prompt for a single scene.

    Requirements:
    - Must describe a complete visual scene
    - Must Include subject(s)(can be humans, animals, etc.), environment, composition
    - Include a clearly identifiable focal subject from this category: {subject_category}
    - Place the scene in this environment type: {environment_type}
    - Must be concrete and visually grounded
    - Avoid abstract storytelling. Avoid vague or high-level descriptions. Focus on specific visual details.
    - Make the subject, setting, objects, and composition distinct from the previous scenes below.

    Focus on specific physical spaces with unique objects and layout.

    {previous_scenes_text}
    
    Blacklist: AVOID prompts related to:
    - two, solitary (or any words indicating number )
    - Top-down view of a cluttered watchmaker’s bench
    - laudromat
    - underground subway station platform, tram
    - chef, violinist...
    - cafe, greenhouse, botanist
    - neon, diver, glass, koi
    - workshop, repair, stormy, beekeeper, lighthouse, control room, library, lab, brass, lobby, classroom, market
    - cavern, pool
    Output 1–2 sentences only. STRICT RULE: Maximum 15 words.
    Output should be text-only, and should be complete sentences(not phrases put together). DO NOT output JSON.
    """
    print(
        "[BaseScene] Sampling category={!r}, environment={!r}; excluding {} recent scene(s).".format(
            subject_category, environment_type, len(previous_scenes)
        ),
        flush=True,
    )
    return call_openai(
        prompt,
        temperature=TEMPERATURE,
        system_prompt=(
            "You create varied, concrete text-to-image scene descriptions. "
            "Return only the requested description."
        ),
    )


def build_edit_prompt(base_scene, edit_types):
    return f"""
You are generating a multi-step image editing trajectory.

Initial image description:
{base_scene}

You must generate a sequence of {len(edit_types)} editing instructions.

Each step has:
- step_id (starting from 0 and incrementing by 1)
- edit_type
- instruction (1 sentence, max 20 words)

edit types in order:
{edit_types}

Rules:
- Step 0's edit type is text_to_image (re-describe full scene)
- local_edit: small localized modification (e.g. add/remove object, change object pose, add small detail)
- global_edit: change lighting, weather, overall environment (e.g. change camera pose, sunny to rainy, indoor to outdoor...)
- text_edit: modify text elements in scene (e.g. signboard, book title, text on clothing)
- style_transfer: change artistic style (e.g. photorealistic to watercolor, anime style, etc.)
- character_edit: modify character appearance or identity (e.g. change the character's gender, look)

Constraints:
- Each step must logically build on previous one
- Maintain scene consistency unless explicitly changed
- Avoid repetition of same transformation in consecutive steps and across different trajectories
- Each step should only introduce one major change to ensure clear progression

Output JSON ONLY:
{{
  "edits": [
    {{
      "step_id": 0,
      "edit_type": "...",
      "instruction": "..."
       
    }},
    ...
  ]
}}
"""


def generate_edit_chain(base_scene):
    num_steps = 5

    edit_types = random.choices(EDIT_TYPES, weights=EDIT_WEIGHTS, k=num_steps - 1)

    edit_types = ["text_to_image"] + edit_types
    prompt = build_edit_prompt(base_scene, edit_types)

    content = call_openai(prompt, temperature=TEMPERATURE, max_output_tokens=1500)
    data = safe_json_load(content)

    if data is None:
        raise ValueError("invalid output")

    return data


def load_existing(save_path):
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError as exc:
                print(f"⚠️ Invalid JSON file: {save_path}")
                raise ValueError(
                    "Cannot continue with an invalid existing benchmark file: {}".format(
                        save_path
                    )
                ) from exc
    return []


def save_dataset(dataset, save_path):
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)


def generate_image_edit_dataset(num_stories=NUM_STORIES, save_path=None, save_every=1):
    if save_path is None:
        save_path = (
            Path(__file__).resolve().parents[2]
            / "test_bench"
            / "image_edit_benchmark.json"
        )
    success_count = 0
    dataset = load_existing(save_path)
    cnt = len(dataset)
    batch_buffer = []
    previous_scenes = [
        task.get("base_prompt", "")
        for task in dataset
        if isinstance(task, dict) and task.get("base_prompt")
    ]

    for i in range(num_stories):
        story_id = f"edit_{cnt:04d}"

        base_scene = generate_base_prompt(previous_scenes)
        print(f"base scene for {story_id}: {base_scene}")
        edits = generate_edit_chain(base_scene)
        task = {"task_id": story_id, "base_prompt": base_scene, "edits": edits["edits"]}
        dataset.append(task)
        batch_buffer.append(task)
        previous_scenes.append(base_scene)
        time.sleep(1)
        cnt += 1
        if len(batch_buffer) >= save_every:
            save_dataset(dataset, save_path)
            print(f"💾 Saved checkpoint ({len(dataset)} total stories)")
            batch_buffer.clear()

    save_dataset(dataset, save_path)

    print("🎉 Done!")
    print(f"Total stories: {len(dataset)}")


if __name__ == "__main__":
    generate_image_edit_dataset()
