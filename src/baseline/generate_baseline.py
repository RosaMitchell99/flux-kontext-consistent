import argparse
import json
from pathlib import Path

import torch
from diffusers import FluxKontextPipeline

MODEL_ID = "black-forest-labs/FLUX.1-Kontext-dev"
GUIDANCE_SCALE = 2.5


def _load_benchmark(path):
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)

    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        if isinstance(payload.get("stories"), list):
            records = payload["stories"]
        elif isinstance(payload.get("tasks"), list):
            records = payload["tasks"]
        elif isinstance(payload.get("edits"), list):
            records = payload["edits"]
        else:
            raise ValueError(
                "Benchmark object must contain 'stories', 'tasks', or 'edits'."
            )
    else:
        raise ValueError("Benchmark JSON root must be a list or object.")

    if not records:
        raise ValueError("Benchmark contains no samples.")
    first = records[0]
    if not isinstance(first, dict):
        raise ValueError("Benchmark samples must be JSON objects.")
    if "task_id" in first or "base_prompt" in first or "edits" in first:
        task_type = "edit"
    elif "story_id" in first or "scenes" in first:
        task_type = "story"
    else:
        raise ValueError("Could not infer task type from benchmark sample fields.")
    return records, task_type


def _generate(pipe, prompt, previous_image=None):
    kwargs = {"prompt": prompt, "guidance_scale": GUIDANCE_SCALE}
    if previous_image is not None:
        kwargs["image"] = previous_image
    with torch.inference_mode():
        return pipe(**kwargs).images[0]


def _generate_story(pipe, stories, output_root):
    for story_index, story in enumerate(stories):
        story_id = str(story.get("story_id", "story_{:04d}".format(story_index)))
        scenes = story.get("scenes")
        if not isinstance(scenes, list):
            print(
                "[Baseline] Skipping {}: scenes is not a list.".format(story_id),
                flush=True,
            )
            continue

        story_output_dir = output_root / story_id
        story_output_dir.mkdir(parents=True, exist_ok=True)
        previous_image = None
        print(
            "\n[Baseline] Story {}: {} scene(s).".format(story_id, len(scenes)),
            flush=True,
        )

        for index, scene in enumerate(scenes):
            instruction = str(scene.get("instruction", "")).strip()
            if not instruction:
                print("[Baseline] Skipping empty scene {}.".format(index), flush=True)
                continue
            print(
                "[Baseline] {}/{} scene {} ({}): {}".format(
                    index + 1,
                    len(scenes),
                    scene.get("scene_id", index),
                    scene.get("generation_type", "story"),
                    instruction,
                ),
                flush=True,
            )
            image = _generate(pipe, instruction, previous_image)
            image_path = story_output_dir / "scene_{}.png".format(index)
            image.save(image_path)
            previous_image = image
            print("[Baseline] Saved {}".format(image_path), flush=True)


def _generate_edit(pipe, tasks, output_root):
    for task_index, task in enumerate(tasks):
        task_id = str(task.get("task_id", "edit_{:04d}".format(task_index)))
        edits = task.get("edits")
        if not isinstance(edits, list):
            print(
                "[Baseline] Skipping {}: edits is not a list.".format(task_id),
                flush=True,
            )
            continue

        task_output_dir = output_root / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)
        previous_image = None
        print(
            "\n[Baseline] Edit task {}: {} step(s).".format(task_id, len(edits)),
            flush=True,
        )

        for index, edit in enumerate(edits):
            instruction = str(edit.get("instruction", "")).strip()
            if not instruction:
                print(
                    "[Baseline] Skipping empty edit step {}.".format(index), flush=True
                )
                continue
            edit_type = str(edit.get("edit_type", "edit"))
            step_id = edit.get("step_id", index)
            print(
                "[Baseline] {}/{} step {} ({}): {}".format(
                    index + 1,
                    len(edits),
                    step_id,
                    edit_type,
                    instruction,
                ),
                flush=True,
            )
            # Later edit steps use the preceding frame as image input.
            image = _generate(pipe, instruction, previous_image)
            image_path = task_output_dir / "step_{}.png".format(step_id)
            image.save(image_path)
            previous_image = image
            print("[Baseline] Saved {}".format(image_path), flush=True)


def _parse_args(argv=None):
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Generate FLUX.1 Kontext baseline outputs for story or edit benchmarks."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        required=True,
        help="Single-story, multi-story, or image-edit benchmark JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "baseline",
        help="Root directory for generated benchmark images.",
    )
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--device", default=None, help="For example: cuda or cpu.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the model from the local Hugging Face cache only.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    benchmark_path = args.benchmark.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    if not benchmark_path.is_file():
        raise FileNotFoundError("Benchmark not found: {}".format(benchmark_path))

    records, task_type = _load_benchmark(benchmark_path)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda"):
        dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
    else:
        dtype = torch.float32
    print(
        "[Baseline] Detected {} benchmark ({} samples); loading {} on {}.".format(
            task_type, len(records), args.model_id, device
        ),
        flush=True,
    )
    pipeline = FluxKontextPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    ).to(device)
    output_root.mkdir(parents=True, exist_ok=True)

    if task_type == "story":
        _generate_story(pipeline, records, output_root)
    else:
        _generate_edit(pipeline, records, output_root)
    print("[Baseline] Finished. Outputs: {}".format(output_root), flush=True)


if __name__ == "__main__":
    main()
