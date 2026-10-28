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

USED_RECENT_ATTRIBUTES = []
MAX_RECENT = 50


def update_and_check_attributes(char_env):

    characters = char_env.get("characters", [])
    if not characters:
        return False

    for c in characters:
        category = c.get("category", "")

        category_counter[category] += 1

        base_attr = c.get("base_attributes", {})

        for k, v in base_attr.items():
            key = f"{k}:{v}"
            attribute_counter[key] += 1

            USED_RECENT_ATTRIBUTES.append(key)

            if attribute_counter[key] > 3:
                return False

    if len(USED_RECENT_ATTRIBUTES) > MAX_RECENT:
        USED_RECENT_ATTRIBUTES[:] = USED_RECENT_ATTRIBUTES[-MAX_RECENT:]

    return True


def call_openai(prompt, temperature=0.8, max_output_tokens=2048):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a JSON generator."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_output_tokens,
    )
    content = response.choices[0].message.content
    content = content.strip().replace("```json", "").replace("```", "")
    return content


NUM_STORIES = 10
SAVE_PATH = "multi_story_benchmark.json"
MIN_SCENES = 3
MAX_SCENES = 6
TEMPERATURE = 0.8
STYLE_POOL = [
    "cinematic",
    "anime style",
    "watercolor painting",
    "photorealistic",
    "fantasy illustration",
]
CATEGORY_POOL = [
    "human",
    "animal",
    "robot",
    "plant",
    "alien",
    "monster",
    "cartoon_character",
    "mythical_creature",
    "object_with_life",
]

CATEGORY_WEIGHTS = [
    8,  # human（最高）
    3,  # animal
    1,  # robot
    3,  # plant
    1,  # alien
    1,  # monster
    1,  # cartoon_character
    1,  # mythical_creature
    1,  # object_with_life
]


def safe_json_load(content):
    try:
        return json.loads(content)
    except:
        return None


def generate_characters_and_environment(story_id, character_ids, categories):
    prompt = f"""
    Generate {len(character_ids)} characters and ONE shared environment for a story benchmark in JSON.
    You have no knowledge of any previous stories. Treat this as a new story.
    You should not add human characteristics to non-human characters. For categories animal and plant, specify the species. Choose the character randomly and do not repeat the same character across stories. 
    Ensure high diversity:
    - Avoid repeating common patterns and attributes across characters and environments
    - Avoid cinematic or dramatic tropes
    - Avoid repeating species
    - DO NOT repeat species or attributes across characters
    Maintain a "used attributes list". Before generating each new character:
    - DO NOT reuse any attribute value (including species) that has appeared in the last 10 samples
    Do not generate decommissioned industrial settings.
    Choose randomly: Environment type MUST be exactly one of:
    - natural_outdoor (nature-dominant outdoor)
    - pure_nature (no man-made objects allowed)
    - pure_indoor (fully enclosed man-made interior)
    - urban_open_space (city space, no nature dominance)
    - domestic_indoor (houses, apartments ONLY)
    - institutional_indoor (schools, hospitals, offices ONLY)
    - fantasy_setting (magic, mythical, sci-fi, etc.)
    expand these patterns creatively into specific settings. Add creativity and style.
    
    - Each character MUST correspond to:
    ids: {character_ids}
    categories: {categories}
    "base_attributes": leave it empty for now.
    - Generate a brief, basic background/environment description for the story, as a string under the key "environment". 1~2 sentences, maximum 20 words.
    Avoid large-scale generic environments like "forest" or "city", or common indoor settings like: "library", "attic", "alley" or "lighthouse", "floating platform","salt flat","lab","corridor","control room","conservatory","lobby","planetarium","vault","gift shop","archive","pool","reading room","office","tram","studio",
    "greenhouse", "workshop","classroom".
    NO "office"!
    Prefer specific settings. The setting does not have to relate to the character's category.
    DO NOT use repeated environments across stories. 

    - Output **valid JSON only** with no extra text
    - Example output format:
    {{
    "story_id": "{story_id}",
    "characters": [
        {{"id": "",
        "category": "",
        "base_attributes": {{}}
        }},
        ...
    ],
    "environment": ...
    }}
    """
    content = call_openai(prompt, temperature=TEMPERATURE)
    data = safe_json_load(content)
    if data is None:
        raise ValueError("Invalid JSON from GPT-4")
    return data


def generate_story_scenes(story_id, char_env, scene_num):
    style = random.choice(STYLE_POOL)
    print(char_env)
    characters_json = json.dumps(char_env["characters"], ensure_ascii=False)
    environment_json = char_env["environment"]

    prompt = f"""
Generate a story benchmark in JSON for story_id: {story_id}
Treat this as a new story. Do NOT reference previous stories. 
Ensure high diversity:
    - Avoid repeating common patterns
    - Avoid cinematic or dramatic tropes
```json
Use the following characters and environment:
Characters: {characters_json}
Environment: {environment_json}
Style: {style}
The background/environment can be used as a reference for scene generation but feel free to expand creatively.
STRICT REQUIREMENT:
Keep all original top-level fields from the input JSON exactly as they are (characters, environment, style). 
Do not remove or modify them. Only add a "scenes" array with the generated scenes.

Requirements:
- Scenes: generate exactly {scene_num} scenes for the story
- scene_id must be an integer starting from 0 and incrementing by 1 for each scene
- First scene generation_type must be: text_to_image, the rest: multi_subject_story_generation
- First scene completely sets up the main characters and environment visually based on the input JSON. It should clearly show ALL main characters and the environment. Describe their detailed visual features respectively.
- Each scene must include:
  - scene_id
  - generation_type
  - instruction (specific visual description of the scene for generation. Keep each instruction short (1-2 sentences, maximum 20 words). 
Focus on visual description and object placement. )
  - state_after (a dictionary of scene and a list of character state for each character. character_id must correspond to the input JSON. character_state MUST include EXACTLY all characters from the input)
- Story must have:
  - Clear narrative arc 
  - Scene transitions (location, time, environment)
  - action (and emotion, but action is primary) changes
  - character interactions (At least 50% of scenes must involve interaction between two or more characters. They can have friendly, neutral or hostile interactions.)
- Avoid low-level image edit instructions (blur, glow, color adjustments)
- Always use explicit noun phrases when referring to characters.
- Use natural noun phrases to refer to them in all instructions. Be sure to distinguish between characters. Do not use their ID or category to refer to them. You can specify their gender/species if relevant(man, woman, etc.).
- Output valid JSON only
-output format:
{{
  "story_id": "<story_id>",
  "characters": <characters_json>,
  "environment": "<environment_json>",
  "style": "<style>",
  "scenes": [
    {{"scene_id": 0,"generation_type": "...","instruction": "...",
    "state_after": {{
        "environment_state": ...,
        "character_state": [{{
        "character_id": ...,
        "action": ...,
        "location": ...,
        "emotion": ...
        }},...]
    }}
    }}, ...
  ]
}}
IMPORTANT:
- Use English ONLY for all text fields.
- Do not insert any other language or symbols.
"""
    content = call_openai(prompt, temperature=TEMPERATURE, max_output_tokens=3000)
    data = safe_json_load(content)
    if data is None:
        raise ValueError("Invalid JSON from GPT-4")
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


def get_start_char_id(dataset):
    max_id = -1
    for story in dataset:
        for char in story.get("characters", []):
            cid = char.get("id", "")
            if cid.startswith("char_"):
                try:
                    num = int(cid.split("_")[1])
                    max_id = max(max_id, num)
                except:
                    continue
    return max_id + 1


def generate_multi_story_dataset(num_stories=NUM_STORIES, save_path=None, save_every=1):
    if save_path is None:
        save_path = (
            Path(__file__).resolve().parents[2]
            / "test_bench"
            / "multi_story_benchmark.json"
        )
    success_count = 0
    dataset = load_existing(save_path)
    cnt = len(load_existing(save_path))
    batch_buffer = []
    global_char_id = get_start_char_id(dataset)

    for i in range(num_stories):
        story_id = f"multi_{cnt+success_count:04d}"
        num_chars = random.choices([2, 3], weights=[0.7, 0.3], k=1)[0]
        character_ids = []
        for _ in range(num_chars):
            character_ids.append(f"char_{global_char_id:06d}")
            global_char_id += 1
        categories = random.choices(
            CATEGORY_POOL, weights=CATEGORY_WEIGHTS, k=num_chars
        )
        try:
            scene_num = random.choices(range(MIN_SCENES, MAX_SCENES + 1), k=1)[0]

            max_retry = 3
            char_env = None

            for r in range(max_retry):
                categories = random.choices(
                    CATEGORY_POOL, weights=CATEGORY_WEIGHTS, k=num_chars
                )
                char_env = generate_characters_and_environment(
                    story_id,
                    character_ids=character_ids,
                    categories=categories,
                )

                if update_and_check_attributes(char_env):
                    break
                else:
                    print(f"⚠️ rejected repetitive sample, retry {r+1}")

                char_env = None

            if char_env is None:
                print(f"❌ Skip {story_id} (no valid char_env)")
                continue

            story_json = generate_story_scenes(story_id, char_env, scene_num)

            dataset.append(story_json)
            batch_buffer.append(story_json)
            success_count += 1
            print(f"✅ Generated {story_id}")

        except Exception as e:
            print(f"❌ Failed {story_id}: {e}")
            continue

        time.sleep(1)

        if len(batch_buffer) >= save_every:
            save_dataset(dataset, save_path)
            print(f"💾 Saved checkpoint ({len(dataset)} total stories)")
            batch_buffer.clear()

    save_dataset(dataset, save_path)

    print("🎉 Done!")
    print(f"Total stories: {len(dataset)}")


if __name__ == "__main__":
    generate_multi_story_dataset()
