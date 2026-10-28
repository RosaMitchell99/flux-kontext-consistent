import argparse
import json
import re
from collections import Counter
from pathlib import Path

STOP_WORDS = set(
    """
    a an the and or but if then than so as at by for from in into of off on onto
    out over through to up with within without about above across after against
    along around before behind below beneath beside between beyond during except
    near since toward towards under until upon via per is am are was were be been
    being have has had do does did can could may might must shall should will
    would it its this that these those there here who whom whose which what when
    where why how all any both each few more most other some such no nor not only
    own same too very just also still already again further once ever never
    someone somebody something anyone anybody anything everyone everybody
    everything nobody nothing he him his she her hers they them their theirs we
    us our ours you your yours i me my mine oneself itself themselves himself
    herself themselves
    """.split()
)

# Ignore common edit and rendering terms when filtering repeated content.
STOP_WORDS.update(
    """
    add adds added adding change changes changed changing modify modifies
    modified modifying replace replaces replaced replacing remove removes
    removed removing delete deletes deleted deleting move moves moved moving
    shift shifts shifted shifting insert inserts inserted inserting put place
    make makes made turn turns turned give gives given keep keeps kept preserve
    preserve preserving alter alters altered edit edits edited editing
    lighting light lights brightly bright brightness dark darker illumination
    illuminate illuminated shadow shadows shading relight relighting exposure
    contrast color colour tone temperature warm cool soft hard ambient
    background foreground image scene photo picture style realistic realism
    unchanged same existing maintain retain retain existing while without
    create transform glowing generate small
    """.split()
)

TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?", re.IGNORECASE)


class BenchmarkFormatError(ValueError):
    pass


def _is_nonempty_string(value):
    return isinstance(value, str) and bool(value.strip())


def _validate_story(story, location):
    if not isinstance(story, dict):
        raise BenchmarkFormatError("{} must be an object".format(location))
    if not _is_nonempty_string(story.get("story_id")):
        raise BenchmarkFormatError(
            "{}.story_id must be a non-empty string".format(location)
        )
    characters = story.get("characters")
    if not isinstance(characters, list) or not characters:
        raise BenchmarkFormatError(
            "{}.characters must be a non-empty list".format(location)
        )
    for index, character in enumerate(characters):
        if not isinstance(character, dict):
            raise BenchmarkFormatError(
                "{}.characters[{}] must be an object".format(location, index)
            )
        attributes = character.get("base_attributes")
        if not isinstance(attributes, dict):
            raise BenchmarkFormatError(
                "{}.characters[{}].base_attributes must be an object".format(
                    location, index
                )
            )
    scenes = story.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise BenchmarkFormatError(
            "{}.scenes must be a non-empty list".format(location)
        )
    for index, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            raise BenchmarkFormatError(
                "{}.scenes[{}] must be an object".format(location, index)
            )
        if not _is_nonempty_string(scene.get("instruction")):
            raise BenchmarkFormatError(
                "{}.scenes[{}].instruction must be a non-empty string".format(
                    location, index
                )
            )
        if "scene_id" in scene and not isinstance(scene["scene_id"], (int, str)):
            raise BenchmarkFormatError(
                "{}.scenes[{}].scene_id must be an integer or string".format(
                    location, index
                )
            )


def _validate_edit_task(task, location):
    if not isinstance(task, dict):
        raise BenchmarkFormatError("{} must be an object".format(location))
    if not _is_nonempty_string(task.get("task_id")):
        raise BenchmarkFormatError(
            "{}.task_id must be a non-empty string".format(location)
        )
    if not _is_nonempty_string(task.get("base_prompt")):
        raise BenchmarkFormatError(
            "{}.base_prompt must be a non-empty string".format(location)
        )
    edits = task.get("edits")
    if not isinstance(edits, list) or not edits:
        raise BenchmarkFormatError("{}.edits must be a non-empty list".format(location))
    for index, edit in enumerate(edits):
        if not isinstance(edit, dict):
            raise BenchmarkFormatError(
                "{}.edits[{}] must be an object".format(location, index)
            )
        if not _is_nonempty_string(edit.get("edit_type")):
            raise BenchmarkFormatError(
                "{}.edits[{}].edit_type must be a non-empty string".format(
                    location, index
                )
            )
        if not _is_nonempty_string(edit.get("instruction")):
            raise BenchmarkFormatError(
                "{}.edits[{}].instruction must be a non-empty string".format(
                    location, index
                )
            )
        if "step_id" in edit and not isinstance(edit["step_id"], (int, str)):
            raise BenchmarkFormatError(
                "{}.edits[{}].step_id must be an integer or string".format(
                    location, index
                )
            )


def _get_samples(payload):
    if isinstance(payload, list):
        return payload, lambda samples: samples
    if not isinstance(payload, dict):
        raise BenchmarkFormatError("JSON root must be a list or object")

    for key in ("stories", "tasks", "edits"):
        if key in payload:
            if not isinstance(payload[key], list):
                raise BenchmarkFormatError("Root field '{}' must be a list".format(key))
            return payload[key], lambda samples, key=key: dict(
                payload, **{key: samples}
            )
    raise BenchmarkFormatError(
        "JSON object must contain 'stories', 'tasks', or 'edits'"
    )


def _infer_kind(samples):
    for sample in samples:
        if isinstance(sample, dict):
            if "task_id" in sample or "edits" in sample:
                return "edit"
            if "story_id" in sample or "scenes" in sample:
                return "story"
    raise BenchmarkFormatError("No story or edit samples found")


def _collect_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _collect_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _collect_strings(child)


def _sample_words(sample, kind):
    if kind == "story":
        text_fields = [sample.get("environment", "")]
        for character in sample.get("characters", []):
            text_fields.extend(
                [
                    character.get("name", ""),
                    character.get("category", ""),
                    character.get("base_attributes", {}),
                ]
            )
        for scene in sample.get("scenes", []):
            text_fields.extend(
                [scene.get("instruction", ""), scene.get("state_after", {})]
            )
    else:
        text_fields = [sample.get("base_prompt", "")]
        for edit in sample.get("edits", []):
            text_fields.extend(
                [
                    edit.get("instruction", ""),
                    edit.get("target_subject", ""),
                    edit.get("target_object", ""),
                    edit.get("target", ""),
                    edit.get("segment_prompt", ""),
                ]
            )

    words = set()
    for text in _collect_strings(text_fields):
        for token in TOKEN_PATTERN.findall(text.casefold()):
            token = token.strip("'")
            if len(token) > 1 and token not in STOP_WORDS and not token.isdigit():
                words.add(token)
    return words


def filter_payload(payload, max_samples_per_word=5):
    samples, wrap = _get_samples(payload)
    kind = _infer_kind(samples)
    counts = Counter()
    accepted = []
    rejected = []
    seen_ids = set()

    for index, sample in enumerate(samples):
        location = "sample[{}]".format(index)
        try:
            if kind == "story":
                _validate_story(sample, location)
                sample_id = sample["story_id"]
            else:
                _validate_edit_task(sample, location)
                sample_id = sample["task_id"]
            if sample_id in seen_ids:
                raise BenchmarkFormatError("duplicate sample id {!r}".format(sample_id))
            words = _sample_words(sample, kind)
            excessive = sorted(
                word for word in words if counts[word] + 1 > max_samples_per_word
            )
            if excessive:
                rejected.append(
                    {
                        "index": index,
                        "sample_id": sample_id,
                        "reason": "word frequency limit exceeded",
                        "words": excessive,
                    }
                )
                continue
        except BenchmarkFormatError as exc:
            rejected.append(
                {"index": index, "reason": "invalid format: {}".format(exc)}
            )
            continue

        accepted.append(sample)
        seen_ids.add(sample_id)
        counts.update(words)

    return wrap(accepted), {
        "kind": kind,
        "input_samples": len(samples),
        "accepted_samples": len(accepted),
        "rejected_samples": len(rejected),
        "max_samples_per_word": max_samples_per_word,
        "word_sample_counts": dict(sorted(counts.items())),
        "rejections": rejected,
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate benchmark JSON and filter repetitive samples."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="A benchmark JSON file or a directory containing benchmark JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path for file input; output directory for directory input.",
    )
    parser.add_argument(
        "--max-samples-per-word",
        type=int,
        default=5,
        help="Maximum accepted samples containing each meaningful word (default: 5).",
    )
    return parser.parse_args(argv)


def _process_file(input_path, output_path, max_samples_per_word):
    with input_path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    filtered, report = filter_payload(payload, max_samples_per_word)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as destination:
        json.dump(filtered, destination, indent=2, ensure_ascii=False)
    report_path = output_path.with_name(output_path.stem + "_filter_report.json")
    with report_path.open("w", encoding="utf-8") as destination:
        json.dump(report, destination, indent=2, ensure_ascii=False)
    print(
        "[BenchmarkFilter] {}: kept {}/{} samples; removed {}. Output: {}".format(
            input_path.name,
            report["accepted_samples"],
            report["input_samples"],
            report["rejected_samples"],
            output_path,
        ),
        flush=True,
    )
    print("[BenchmarkFilter] Filter report: {}".format(report_path), flush=True)


def main(argv=None):
    args = _parse_args(argv)
    input_path = args.input.expanduser().resolve()
    if args.max_samples_per_word < 1:
        raise ValueError("--max-samples-per-word must be at least 1")
    if not input_path.exists():
        raise FileNotFoundError("Input not found: {}".format(input_path))

    if input_path.is_file():
        output_path = (
            args.output.expanduser().resolve()
            if args.output
            else input_path.with_name(input_path.stem + "_filtered.json")
        )
        _process_file(input_path, output_path, args.max_samples_per_word)
        return

    output_dir = (
        args.output.expanduser().resolve() if args.output else input_path / "filtered"
    )
    json_files = sorted(
        path
        for path in input_path.rglob("*.json")
        if output_dir not in path.parents
        and path != output_dir
        and not path.stem.endswith("_filtered")
        and not path.stem.endswith("_filter_report")
    )
    if not json_files:
        raise FileNotFoundError(
            "No benchmark JSON files found in {}".format(input_path)
        )
    for source_path in json_files:
        relative = source_path.relative_to(input_path)
        output_path = output_dir / relative.parent / (relative.stem + "_filtered.json")
        try:
            _process_file(source_path, output_path, args.max_samples_per_word)
        except (OSError, json.JSONDecodeError, BenchmarkFormatError) as exc:
            print(
                "[BenchmarkFilter] Skipped {}: {}".format(source_path, exc),
                flush=True,
            )


if __name__ == "__main__":
    main()
