import json
import os
import random
import re
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


NUM_STORIES = 50
SAVE_PATH = "single_story_benchmark.json"
MIN_SCENES = 5
MAX_SCENES = 10
TEMPERATURE = 0.9
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

ENVIRONMENT_TYPE_POOL = [
    "natural_outdoor",
    "pure_nature",
    "pure_indoor",
    "urban_open_space",
    "domestic_indoor",
    "institutional_indoor",
    "fantasy_setting",
]
ENVIRONMENT_TYPE_WEIGHTS = [1] * len(ENVIRONMENT_TYPE_POOL)
COMMON_SETTING_BLACKLIST = (
    "workshop",
    "lighthouse",
    "underwater",
    "market",
    "station",
    "railway",
    "subway",
    "depot",
    "annex",
    "office",
    "archive",
    "greenhouse",
)


def find_common_settings(story):
    text_parts = [str(story.get("environment", ""))]
    for scene in story.get("scenes", []):
        if not isinstance(scene, dict):
            continue
        text_parts.append(str(scene.get("instruction", "")))
        state_after = scene.get("state_after", {})
        environment_state = (
            state_after.get("environment_state", {})
            if isinstance(state_after, dict)
            else {}
        )
        text_parts.append(json.dumps(environment_state, ensure_ascii=False))
    text = " ".join(text_parts)
    return [
        term
        for term in COMMON_SETTING_BLACKLIST
        if re.search(r"\b{}\b".format(re.escape(term)), text, re.IGNORECASE)
    ]


def safe_json_load(content):
    try:
        return json.loads(content)
    except:
        return None


def generate_characters_and_environment(
    story_id, character_id, category, environment_type
):
    prompt = f"""
    Generate a character and environment configuration for a story benchmark in JSON.
    You have no knowledge of any previous stories. Treat this as a new story. Do NOT reference previous stories.
    You should not add human characteristics to non-human characters. For categories animal and plant, specify the species. Choose the character randomly and do not repeat the same character across stories. 
    Ensure high diversity:
    - Avoid repeating common patterns and attributes across characters and environments
    - Avoid cinematic or dramatic tropes
    - Avoid repeating species
    Maintain a "used attributes list". Before generating each new character:
    - Check previous outputs
    - DO NOT reuse any attribute value (including species) that has appeared in the last 10 samples
    Avoid large-scale generic environments like "forest" or "city".
    Generate a specific, localized setting in this sampled environment type: {environment_type}.
    Environment types: natural_outdoor (nature-dominant outdoor), pure_nature (no man-made objects),
    pure_indoor (fully enclosed man-made interior), urban_open_space (city space without nature dominance),
    domestic_indoor (house or apartment), institutional_indoor (school or hospital),
    fantasy_setting (magic, mythical, or science-fiction setting).
    The environment does not need to be directly related to the character's category. Be creative and diverse in your combinations!
    - You MUST use the following character_id: {character_id}
    - You MUST use the following character category: {category}
    - Each character must include the following fields:
    - "id": a unique character identifier
    - "category": the character's type
    - "base_attributes": leave it empty for now.
    - Generate a brief, basic background/environment description for the story, as a string under the key "environment". 1~2 sentences, maximum 20 words.
    - Avoid these common settings: {", ".join(COMMON_SETTING_BLACKLIST)}
    
    - Output **valid JSON only** with no extra text
    - Example output format:
    {{
    "story_id": "{story_id}",
    "characters": [
        {{"id": "",
        "category": "",
        "base_attributes": {{}}
        }}
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
- First scene generation_type must be: text_to_image, the rest: single_subject_story_generation
- First scene completely sets up the main character and environment visually based on the input JSON. It should clearly show the character and the environment.
- Each scene must include:
  - scene_id
  - generation_type
  - instruction (specific visual description of the scene for generation. Keep each instruction short (1-2 sentences, maximum 20 words). 
Focus on visual description and object placement. )
  - state_after (a dictionary of scene and character state)
- Story must have:
  - Clear narrative arc
  - Scene transitions (location, time, environment)
  - action (and emotion, but action is primary) changes
- Keep the main character consistent across scenes
- Avoid low-level image edit instructions (blur, glow, color adjustments)
- Use a natural noun phrase to refer to them in all instructions and state_after fields. Do not use its ID or category. You can specify its gender/species if relevant(man, woman, etc.)
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
        "character_state": {{
        "action": ...,
        "location": ...,
        "emotion": ...
        }}
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
    if not os.path.exists(save_path):
        return []

    try:
        with open(save_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"⚠️ Invalid JSON file: {save_path}")
        return []


def save_dataset(dataset, save_path):
    directory = os.path.dirname(save_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"💾 Saved {len(dataset)} stories to {save_path}")


def get_next_story_index(dataset, prefix):
    max_index = -1
    for story in dataset:
        story_id = str(story.get("story_id", ""))
        if not story_id.startswith(prefix):
            continue
        try:
            max_index = max(max_index, int(story_id[len(prefix) :]))
        except ValueError:
            continue
    return max_index + 1


def get_next_character_index(dataset):
    max_index = -1
    for story in dataset:
        for character in story.get("characters", []):
            character_id = str(character.get("id", ""))
            if not character_id.startswith("char_"):
                continue
            try:
                max_index = max(max_index, int(character_id[len("char_") :]))
            except ValueError:
                continue
    return max_index + 1


def generate_single_story_dataset(
    num_stories=NUM_STORIES, save_path=None, save_every=1
):
    if save_path is None:
        save_path = (
            Path(__file__).resolve().parents[2]
            / "test_bench"
            / "single_story_benchmark.json"
        )
    dataset = load_existing(save_path)
    next_story_index = get_next_story_index(dataset, "single_")
    next_character_index = get_next_character_index(dataset)
    batch_buffer = []

    print(
        f"📂 Loaded {len(dataset)} existing stories; next ID is single_{next_story_index:04d}; "
        f"next character ID is char_{next_character_index:06d}"
    )

    for i in range(num_stories):
        story_id = f"single_{next_story_index:04d}"
        character_id = f"char_{next_character_index:06d}"
        environment_type = random.choices(
            ENVIRONMENT_TYPE_POOL,
            weights=ENVIRONMENT_TYPE_WEIGHTS,
            k=1,
        )[0]
        print(f"🎲 Sampled environment type for {story_id}: {environment_type}")

        try:
            scene_num = random.choices(range(MIN_SCENES, MAX_SCENES + 1), k=1)[0]

            max_retry = 3
            char_env = None

            for r in range(max_retry):
                category = random.choices(CATEGORY_POOL, weights=CATEGORY_WEIGHTS, k=1)[
                    0
                ]
                char_env = generate_characters_and_environment(
                    story_id,
                    character_id,
                    category,
                    environment_type,
                )

                blocked_settings = find_common_settings(char_env)
                if not blocked_settings and update_and_check_attributes(char_env):
                    break
                else:
                    reason = (
                        "common setting(s): " + ", ".join(blocked_settings)
                        if blocked_settings
                        else "repetitive character attributes"
                    )
                    print(f"⚠️ rejected {story_id} ({reason}), retry {r+1}")

                char_env = None

            if char_env is None:
                print(f"❌ Skip {story_id} (no valid char_env)")
                continue

            story_json = None
            for scene_retry in range(3):
                candidate = generate_story_scenes(story_id, char_env, scene_num)
                blocked_settings = find_common_settings(candidate)
                if not blocked_settings:
                    story_json = candidate
                    break
                print(
                    f"⚠️ rejected {story_id} story for common setting(s): "
                    f"{', '.join(blocked_settings)}, retry {scene_retry + 1}"
                )
            if story_json is None:
                print(f"❌ Skip {story_id} (story repeatedly used a common setting)")
                continue

            dataset.append(story_json)
            batch_buffer.append(story_json)
            next_story_index += 1
            next_character_index += len(story_json.get("characters", []))
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
    generate_single_story_dataset()
