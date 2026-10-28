import argparse
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    # Support direct script execution and module execution.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from agent.controller import AgentController
    from agent.generator import Generator
    from agent.memory import MemorySystem
    from agent.verifier import Verifier
else:
    from .controller import AgentController
    from .generator import Generator
    from .memory import MemorySystem
    from .verifier import Verifier


def _parse_args(argv=None):
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Generate Agent results for story or image-edit benchmarks."
    )
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=project_root / "test_bench" / "single_story_benchmark.json",
        help="Benchmark JSON containing stories or edit tasks.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Root directory for generated images and result manifests.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        help="Maximum scenes/steps to process per story or edit task.",
    )
    parser.add_argument(
        "--memory-file",
        type=Path,
        help="Persistent Agent memory JSON (default: <output-dir>/agent_memory.json).",
    )
    parser.add_argument(
        "--model-id",
        default=Generator.MODEL_ID,
        help="FLUX.1 Kontext Hugging Face model ID or local model directory.",
    )
    parser.add_argument(
        "--clip-model",
        default=Verifier.MODEL_ID,
        help="CLIP Hugging Face model ID or local model directory.",
    )
    return parser.parse_args(argv)


def _load_benchmark(path):
    with path.open("r", encoding="utf-8") as benchmark_file:
        payload = json.load(benchmark_file)

    if isinstance(payload, list):
        records = payload
        root_key = None
    elif isinstance(payload, dict):
        root_key = next(
            (key for key in ("stories", "tasks", "edits") if key in payload),
            None,
        )
        if root_key is None or not isinstance(payload[root_key], list):
            raise ValueError(
                "Benchmark JSON must be a list or contain a list of stories/tasks/edits."
            )
        records = payload[root_key]
    else:
        raise ValueError("Benchmark JSON root must be a list or object.")

    if not records:
        raise ValueError("Benchmark contains no samples.")
    first = records[0]
    if not isinstance(first, dict):
        raise ValueError("Benchmark samples must be objects.")
    if root_key in ("tasks", "edits") or any(
        key in first for key in ("task_id", "base_prompt", "edits")
    ):
        task_type = "edit"
    elif root_key == "stories" or any(key in first for key in ("story_id", "scenes")):
        task_type = "story"
    else:
        raise ValueError("Could not infer benchmark task type from its fields.")
    return records, task_type


def _verification_summary(report):
    if not isinstance(report, dict):
        return {}
    return {
        "available": report.get("available", False),
        "passed": report.get("passed"),
        "scores": report.get("scores", {}),
        "feedback": report.get("feedback", []),
        "error": report.get("error"),
    }


def _memory_snapshot(memory):
    return {
        "short_term_turns": [item.get("turn") for item in memory.short_term_memory],
        "long_term_subjects": [
            {"subject_id": key, "name": value.get("name")}
            for key, value in memory.long_term_memory.items()
        ],
    }


def _run_turn(
    controller,
    instruction,
    turn,
    label,
    reference_image=None,
    published_filename=None,
    generation_type=None,
    item_id=None,
    refine_prompt=True,
):
    memory_before = _memory_snapshot(controller.memory)
    print(
        "\n========== {} turn {}{} ==========".format(
            label, turn + 1, " ({})".format(item_id) if item_id else ""
        ),
        flush=True,
    )
    if generation_type:
        print("Generation type: {}".format(generation_type), flush=True)
    print("Instruction: {}".format(instruction), flush=True)
    print("Memory before: {}".format(memory_before), flush=True)

    image_path = controller.process(
        instruction,
        turn,
        reference_image_override=reference_image,
        published_filename=published_filename,
        refine_prompt=refine_prompt,
    )
    debug = controller.last_run or {}
    initial_verification = _verification_summary(debug.get("initial_verification"))
    retry_verification = _verification_summary(debug.get("retry_verification"))
    final_verification = _verification_summary(debug.get("final_verification"))

    print("Subjects: {}".format(debug.get("subjects", [])), flush=True)
    print("Known subject IDs: {}".format(debug.get("known_subjects", [])), flush=True)
    print("New subject IDs: {}".format(debug.get("new_subjects", [])), flush=True)
    print("Reference image: {}".format(debug.get("reference_image")), flush=True)
    print("Refined prompt: {}".format(debug.get("refined_prompt", "")), flush=True)
    print("Verification: {}".format(final_verification), flush=True)
    if debug.get("retry_image"):
        print("Retry image: {}".format(debug["retry_image"]), flush=True)
        print("Retry verification: {}".format(retry_verification), flush=True)
    print("Selected image: {}".format(image_path), flush=True)
    memory_after = _memory_snapshot(controller.memory)

    result = {
        "turn": turn,
        "instruction": instruction,
        "refined_prompt": debug.get("refined_prompt"),
        "subjects": debug.get("subjects", []),
        "known_subject_ids": debug.get("known_subjects", []),
        "new_subject_ids": debug.get("new_subjects", []),
        "reference_image": debug.get("reference_image"),
        "initial_image": debug.get("initial_image"),
        "initial_verification": initial_verification,
        "retry_prompt": debug.get("retry_prompt"),
        "retry_image": debug.get("retry_image"),
        "retry_verification": retry_verification,
        "final_verification": final_verification,
        "generated_image": image_path,
        "memory_before": memory_before,
        "memory_after": memory_after,
    }
    return image_path, result


def _write_results(path, payload):
    with path.open("w", encoding="utf-8") as result_file:
        json.dump(payload, result_file, indent=2, ensure_ascii=False)


def _count_generated_images(item_dir, task_type):
    if not item_dir.is_dir():
        return 0
    pattern = re.compile(
        r"^scene_\d+\.png$" if task_type == "story" else r"^step_.+\.png$"
    )
    return sum(
        1 for path in item_dir.iterdir() if path.is_file() and pattern.match(path.name)
    )


def _expected_image_count(record, task_type, max_scenes):
    entries = record.get("scenes" if task_type == "story" else "edits", [])
    if not isinstance(entries, list):
        return 0
    return min(len(entries), max_scenes) if max_scenes is not None else len(entries)


def _item_output_dir(record, output_root, task_type, index):
    id_field = "story_id" if task_type == "story" else "task_id"
    fallback = (
        "story_{:04d}".format(index)
        if task_type == "story"
        else "edit_{:04d}".format(index)
    )
    return output_root / str(record.get(id_field, fallback))


def _run_story(
    story,
    output_root,
    max_scenes,
    story_index,
    memory,
    memory_path,
    generator,
    clip_model_id,
):
    story_id = str(story.get("story_id", "story_{:04d}".format(story_index)))
    scenes = story.get("scenes", [])
    if not isinstance(scenes, list):
        print("Skipping {}: 'scenes' is not a list.".format(story_id), flush=True)
        return
    if max_scenes is not None:
        scenes = scenes[:max_scenes]

    output_dir = output_root / story_id
    output_dir.mkdir(parents=True, exist_ok=True)
    generator.output_dir = output_dir
    controller = AgentController(
        memory=memory,
        generator=generator,
        verifier=Verifier(model_id=clip_model_id),
        story_characters=story.get("characters", []),
    )
    results = []
    print(
        "\n========== Story {} ({} scenes; characters={}) ==========".format(
            story_id, len(scenes), len(story.get("characters", []))
        ),
        flush=True,
    )

    for turn, scene in enumerate(scenes):
        instruction = str(scene.get("instruction") or "").strip()
        if not instruction:
            print("Skipping scene {}: empty instruction.".format(turn), flush=True)
            continue
        image_path, result = _run_turn(
            controller,
            instruction,
            turn,
            "Story scene {}".format(scene.get("scene_id", turn)),
            generation_type=scene.get("generation_type"),
            item_id=story_id,
        )
        result.update(
            {
                "scene_id": scene.get("scene_id", turn),
                "generation_type": scene.get("generation_type"),
            }
        )
        results.append(result)
        _write_results(
            output_dir / "agent_results.json",
            {
                "story_id": story_id,
                "environment": story.get("environment"),
                "style": story.get("style"),
                "characters": story.get("characters", []),
                "scenes": results,
            },
        )
        memory.save(memory_path)
    print("Completed story {}. Outputs: {}".format(story_id, output_dir), flush=True)


def _run_edit_task(
    task,
    output_root,
    max_scenes,
    task_index,
    memory,
    memory_path,
    generator,
    clip_model_id,
):
    task_id = str(task.get("task_id", "edit_{:04d}".format(task_index)))
    edits = task.get("edits", [])
    if not isinstance(edits, list):
        print("Skipping {}: 'edits' is not a list.".format(task_id), flush=True)
        return
    if max_scenes is not None:
        edits = edits[:max_scenes]

    output_dir = output_root / task_id
    output_dir.mkdir(parents=True, exist_ok=True)
    generator.output_dir = output_dir
    # Continue each edit from its preceding frame.
    controller = AgentController(
        memory=memory,
        generator=generator,
        verifier=Verifier(model_id=clip_model_id),
    )
    results = []
    previous_image = None
    print(
        "\n========== Edit task {} ({} steps) ==========".format(task_id, len(edits)),
        flush=True,
    )

    for turn, edit in enumerate(edits):
        instruction = str(edit.get("instruction") or "").strip()
        if not instruction:
            print("Skipping edit step {}: empty instruction.".format(turn), flush=True)
            continue
        step_id = edit.get("step_id", turn)
        edit_type = edit.get("edit_type", "edit")
        image_path, result = _run_turn(
            controller,
            instruction,
            turn,
            "Edit step {} ({})".format(step_id, edit_type),
            reference_image=previous_image,
            published_filename="step_{}.png".format(step_id),
            generation_type=edit_type,
            item_id=task_id,
            refine_prompt=False,
        )
        result.update(
            {
                "step_id": step_id,
                "edit_type": edit_type,
            }
        )
        results.append(result)
        previous_image = image_path
        _write_results(
            output_dir / "agent_results.json",
            {
                "task_id": task_id,
                "base_prompt": task.get("base_prompt"),
                "edits": edits,
                "scenes": results,
                "steps": results,
            },
        )
        memory.save(memory_path)
    print("Completed edit task {}. Outputs: {}".format(task_id, output_dir), flush=True)


def run_benchmark(
    benchmark_path,
    output_root,
    memory_path,
    max_scenes=None,
    model_id=Generator.MODEL_ID,
    clip_model_id=Verifier.MODEL_ID,
):
    records, task_type = _load_benchmark(benchmark_path)
    memory = MemorySystem.load(memory_path)
    complete = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            complete.append(False)
            continue
        item_dir = _item_output_dir(record, output_root, task_type, index)
        expected = _expected_image_count(record, task_type, max_scenes)
        existing = _count_generated_images(item_dir, task_type)
        complete.append(expected > 0 and existing >= expected)
    # Reuse one pipeline for all benchmark items.
    generator = Generator(output_dir=output_root, model_id=model_id)
    print(
        "[Agent] Detected {} benchmark with {} item(s); {} already complete; "
        "loaded {} long-term subject(s) from {}.".format(
            task_type,
            len(records),
            sum(complete),
            len(memory.long_term_memory),
            memory_path,
        ),
        flush=True,
    )
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            print("Skipping sample {}: expected an object.".format(index), flush=True)
            continue
        # Preserve long-term memory while isolating short-term continuity.
        memory.clear_short_term()
        item_dir = _item_output_dir(record, output_root, task_type, index)
        expected = _expected_image_count(record, task_type, max_scenes)
        existing = _count_generated_images(item_dir, task_type)
        if complete[index]:
            id_field = "story_id" if task_type == "story" else "task_id"
            identifier = record.get(id_field, "sample_{:04d}".format(index))
            print(
                "[Agent] Skipping complete {} {}: {}/{} frame images.".format(
                    task_type, identifier, existing, expected
                ),
                flush=True,
            )
            continue

        id_field = "story_id" if task_type == "story" else "task_id"
        print(
            "[Agent] {} {} is incomplete ({}/{} frame images); restarting it from its first turn.".format(
                task_type, record.get(id_field, index), existing, expected
            ),
            flush=True,
        )
        if task_type == "story":
            _run_story(
                record,
                output_root,
                max_scenes,
                index,
                memory,
                memory_path,
                generator,
                clip_model_id,
            )
        else:
            _run_edit_task(
                record,
                output_root,
                max_scenes,
                index,
                memory,
                memory_path,
                generator,
                clip_model_id,
            )


def main(argv=None):
    args = _parse_args(argv)
    benchmark_path = args.benchmark.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    memory_path = (
        args.memory_file.expanduser().resolve()
        if args.memory_file
        else output_root / "agent_memory.json"
    )
    if not benchmark_path.is_file():
        raise FileNotFoundError("Benchmark file not found: {}".format(benchmark_path))
    output_root.mkdir(parents=True, exist_ok=True)
    if args.max_scenes is not None and args.max_scenes < 1:
        raise ValueError("--max-scenes must be at least 1")
    run_benchmark(
        benchmark_path,
        output_root,
        memory_path=memory_path,
        max_scenes=args.max_scenes,
        model_id=args.model_id,
        clip_model_id=args.clip_model,
    )


if __name__ == "__main__":
    main()
