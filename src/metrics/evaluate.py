import argparse
import json
import re
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metrics.models import (
        SAM3Segmenter,
        VisionMetrics,
        masked_full_image,
        masked_region_crop,
    )
else:
    from .models import (
        SAM3Segmenter,
        VisionMetrics,
        masked_full_image,
        masked_region_crop,
    )


def _load_records(path, task):
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    if isinstance(payload, dict):
        if task == "edit":
            records = payload.get("tasks", payload.get("edits", []))
        else:
            records = payload.get("stories", [])
    else:
        records = payload
    if not isinstance(records, list):
        raise ValueError("Benchmark must be a list or contain a list of stories/tasks.")
    return records


def _detect_task(records):
    if records and ("task_id" in records[0] or "edits" in records[0]):
        return "edit"
    return "story"


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _resolve_image(results_dir, item_id, index, task, manifest_paths=None):
    if manifest_paths and index in manifest_paths:
        recorded = Path(str(manifest_paths[index]))
        if recorded.is_file():
            return recorded
        for candidate in (
            results_dir / item_id / recorded.name,
            results_dir / recorded.name,
        ):
            if candidate.is_file():
                return candidate

    item_dir = results_dir / item_id
    candidate_dirs = [item_dir, results_dir]
    if task == "story":
        names = ("scene_{}.png".format(index), "{}.png".format(index))
    else:
        names = (
            "step_{}.png".format(index),
            "scene_{}.png".format(index),
            "frame_{}.png".format(index),
            "edit_{}.png".format(index),
            "{}.png".format(index),
        )
    for directory in candidate_dirs:
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def _manifest_scene_paths(results_dir, item_id):
    manifest = results_dir / item_id / "agent_results.json"
    if not manifest.is_file():
        return {}
    try:
        with manifest.open("r", encoding="utf-8") as source:
            payload = json.load(source)
    except (OSError, json.JSONDecodeError):
        return {}
    paths = {}
    for fallback_index, scene in enumerate(payload.get("scenes", [])):
        index = scene.get("turn", fallback_index)
        image_path = scene.get("generated_image")
        if image_path:
            paths[index] = image_path
    return paths


def _edit_target_phrase(edit):
    for key in (
        "segment_prompt",
        "target_subject",
        "target_object",
        "target",
        "object",
    ):
        value = edit.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    instruction = str(edit.get("instruction", "")).strip()
    match = re.search(
        r"\b(?:add|place|put|insert|remove|delete|move|shift|change|modify|replace|make|turn|give)\s+"
        r"(?:(?:a|an|the|some|his|her|their)\s+)?(.+)",
        instruction,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    phrase = match.group(1)
    phrase = re.split(
        r"[,;.]|\b(?:at|in|on|to|from|while|showing|keeping|with|but|so that)\b",
        phrase,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    phrase = re.sub(
        r"\s+(?:middle[- ]aged|older|younger|taller|shorter|more|less|with|wearing)\b.*$",
        "",
        phrase,
        flags=re.IGNORECASE,
    )
    return phrase.strip(" \t\n\r'\"-:")


def evaluate_story(records, results_dir, metrics):
    outputs = []
    for story in records:
        story_id = str(story.get("story_id", "story"))
        manifest_paths = _manifest_scene_paths(results_dir, story_id)
        scenes = story.get("scenes", [])
        image_paths = [
            _resolve_image(results_dir, story_id, index, "story", manifest_paths)
            for index in range(len(scenes))
        ]
        clip_i_scores, dino_scores, clip_t_scores = [], [], []
        missing_images = []
        for index, scene in enumerate(scenes):
            current = image_paths[index]
            if current is None:
                missing_images.append(index)
                continue
            instruction = str(scene.get("instruction", "")).strip()
            if instruction:
                clip_t_scores.append(metrics.clip_t(current, instruction))
            if index > 0 and image_paths[index - 1] is not None:
                clip_i_scores.append(metrics.clip_i(image_paths[index - 1], current))
                dino_scores.append(
                    metrics.dino_similarity(image_paths[index - 1], current)
                )
        outputs.append(
            {
                "story_id": story_id,
                "num_scenes": len(scenes),
                "num_images_found": sum(path is not None for path in image_paths),
                "missing_scene_indices": missing_images,
                "metrics": {
                    "clip_i_adjacent": _mean(clip_i_scores),
                    "dino_adjacent": _mean(dino_scores),
                    "clip_t_instruction": _mean(clip_t_scores),
                },
            }
        )
        print(
            "[Metrics] Evaluated story {}: {} image(s) found.".format(
                story_id, len(scenes) - len(missing_images)
            ),
            flush=True,
        )
    metric_names = ("clip_i_adjacent", "dino_adjacent", "clip_t_instruction")
    summary = {
        name: _mean(item["metrics"][name] for item in outputs) for name in metric_names
    }
    return {
        "task": "story",
        "num_items": len(outputs),
        "metrics": summary,
        "items": outputs,
    }


def evaluate_edit(records, results_dir, metrics, sam3_model_id, save_masks_dir=None):
    outputs = []
    segmenter = None
    for task in records:
        task_id = str(task.get("task_id", "edit"))
        edits = task.get("edits", [])
        manifest_paths = _manifest_scene_paths(results_dir, task_id)
        image_paths = [
            _resolve_image(results_dir, task_id, index, "edit", manifest_paths)
            for index in range(len(edits))
        ]
        overall_clip_i, overall_clip_t = [], []
        local_clip_t, outside_clip_i = [], []
        step_results = []

        for index, edit in enumerate(edits):
            current = image_paths[index]
            if current is None:
                step_results.append(
                    {"step_id": index, "error": "Generated image not found."}
                )
                continue
            instruction = str(edit.get("instruction", "")).strip()
            edit_type = str(edit.get("edit_type", "")).casefold()
            step_result = {
                "step_id": edit.get("step_id", index),
                "edit_type": edit_type,
            }
            if instruction:
                score = metrics.clip_t(current, instruction)
                overall_clip_t.append(score)
                step_result["clip_t"] = score

            previous = image_paths[index - 1] if index > 0 else None
            if previous is not None:
                score = metrics.clip_i(previous, current)
                overall_clip_i.append(score)
                step_result["clip_i"] = score

            if edit_type in ("local_edit", "character_edit") and previous is not None:
                target = _edit_target_phrase(edit)
                step_result["sam3_prompt"] = target
                try:
                    if segmenter is None:
                        segmenter = SAM3Segmenter(
                            model_id=sam3_model_id, device=metrics.device
                        )
                    current_mask, confidence = segmenter.segment(current, target)
                    if current_mask is None:
                        step_result["local_error"] = "SAM3 returned no target mask."
                    else:
                        current_image = VisionMetrics.open_image(current)
                        target_region = masked_region_crop(current_image, current_mask)
                        local_score = metrics.clip_t(target_region, instruction)
                        local_clip_t.append(local_score)
                        step_result["local_clip_t_masked_subject"] = local_score
                        step_result["sam3_confidence"] = confidence
                        step_result["mask_area_ratio"] = float(current_mask.mean())

                        previous_mask, _ = segmenter.segment(previous, target)
                        if previous_mask is None:
                            previous_mask = current_mask & False
                        previous_outside = masked_full_image(
                            previous, previous_mask, include_mask=False
                        )
                        current_outside = masked_full_image(
                            current, current_mask, include_mask=False
                        )
                        outside_score = metrics.clip_i(
                            previous_outside, current_outside
                        )
                        outside_clip_i.append(outside_score)
                        step_result["outside_clip_i_unedited_region"] = outside_score

                        if save_masks_dir is not None:
                            from PIL import Image

                            save_masks_dir.mkdir(parents=True, exist_ok=True)
                            mask_path = save_masks_dir / "{}_step_{}.png".format(
                                task_id, index
                            )
                            Image.fromarray(current_mask.astype("uint8") * 255).save(
                                mask_path
                            )
                            step_result["sam3_mask_path"] = str(mask_path)
                except Exception as exc:
                    step_result["local_error"] = str(exc)

            step_results.append(step_result)

        outputs.append(
            {
                "task_id": task_id,
                "num_steps": len(edits),
                "num_images_found": sum(path is not None for path in image_paths),
                "metrics": {
                    "clip_i_overall_adjacent": _mean(overall_clip_i),
                    "clip_t_overall_instruction": _mean(overall_clip_t),
                    "clip_t_local_masked_subject": _mean(local_clip_t),
                    "clip_i_outside_mask": _mean(outside_clip_i),
                },
                "steps": step_results,
            }
        )
        print(
            "[Metrics] Evaluated edit task {}: {} image(s) found.".format(
                task_id, len(edits) - sum(path is None for path in image_paths)
            ),
            flush=True,
        )

    metric_names = (
        "clip_i_overall_adjacent",
        "clip_t_overall_instruction",
        "clip_t_local_masked_subject",
        "clip_i_outside_mask",
    )
    summary = {
        name: _mean(item["metrics"][name] for item in outputs) for name in metric_names
    }
    return {
        "task": "edit",
        "num_items": len(outputs),
        "metrics": summary,
        "items": outputs,
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate story/edit benchmark images using thesis metrics."
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--task", choices=("auto", "story", "edit"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("metrics_results.json"))
    parser.add_argument("--device", default=None, help="For example: cuda or cpu.")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--dino-model", default="facebook/dinov2-base")
    parser.add_argument("--sam3-model", default="facebook/sam3")
    parser.add_argument(
        "--save-masks-dir",
        type=Path,
        help="Optional directory for SAM3 mask PNGs (edit tasks only).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    benchmark_path = args.benchmark.expanduser().resolve()
    results_dir = args.results_dir.expanduser().resolve()
    if not benchmark_path.is_file():
        raise FileNotFoundError("Benchmark not found: {}".format(benchmark_path))
    if not results_dir.is_dir():
        raise NotADirectoryError("Results directory not found: {}".format(results_dir))

    with benchmark_path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    task = args.task
    if task == "auto":
        items = (
            payload.get("tasks", payload.get("edits", []))
            if isinstance(payload, dict)
            else payload
        )
        task = _detect_task(items)
    records = _load_records(benchmark_path, task)
    vision_metrics = VisionMetrics(
        device=args.device,
        clip_model_id=args.clip_model,
        dino_model_id=args.dino_model,
    )
    if task == "story":
        report = evaluate_story(records, results_dir, vision_metrics)
    else:
        mask_dir = (
            args.save_masks_dir.expanduser().resolve() if args.save_masks_dir else None
        )
        report = evaluate_edit(
            records,
            results_dir,
            vision_metrics,
            sam3_model_id=args.sam3_model,
            save_masks_dir=mask_dir,
        )

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2, ensure_ascii=False)
    print("[Metrics] Summary: {}".format(report["metrics"]), flush=True)
    print("[Metrics] Saved report: {}".format(output_path), flush=True)


if __name__ == "__main__":
    main()
