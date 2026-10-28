import json
import math
from pathlib import Path
from uuid import uuid4

from .generator import Generator
from .memory import MemorySystem
from .prompt_refiner import PromptRefiner
from .verifier import Verifier


class AgentController:
    def __init__(
        self,
        memory=None,
        refiner=None,
        generator=None,
        verifier=None,
        story_characters=None,
    ):
        self.memory = memory or MemorySystem(short_term_size=1)
        self.refiner = refiner or PromptRefiner()
        self.generator = generator or Generator()
        self.verifier = verifier or Verifier()
        self.story_characters = story_characters or []
        self.last_run = None
        matcher = getattr(self.verifier, "subject_text_similarity", None)
        if getattr(self.memory, "semantic_matcher", None) is None and matcher:
            self.memory.semantic_matcher = matcher

    @staticmethod
    def _identify_subjects(prompt, story_characters=None):
        print("[Controller] Extracting subjects from prompt...", flush=True)
        try:
            from .llm_utils import call_openai

            character_roster = []
            for character in story_characters or []:
                attributes = character["base_attributes"]
                character_roster.append(
                    {
                        "id": character["id"],
                        "name": attributes.get("species") or character["category"],
                        "category": character["category"],
                        "attributes": attributes,
                    }
                )
            roster_text = json.dumps(character_roster, ensure_ascii=False)
            if character_roster:
                scope_instruction = (
                    "Treat the following story character roster as the only valid subjects. "
                    "Return every roster character that is explicitly present in this "
                    "scene, including characters that are present but not the main "
                    "visual focus. Do not return roster characters absent from the "
                    "scene. Never merge two distinct roster characters into one entry; "
                    "return each present character exactly once. Do not return setting "
                    "details, props, weather, or incidental background entities. Match "
                    "each character to the roster and return its id as subject_id. Use a specific "
                    "natural noun phrase from the scene as name; do not use a generic "
                    "category such as 'human' or 'animal' as the name when the scene "
                    "provides a more specific identity.\n"
                    "Story character roster: {}\n"
                ).format(roster_text)
            else:
                scope_instruction = (
                    "Identify every clearly depicted or mentioned character or animal, "
                    "even when it is not the main focus or has no explicit action. "
                    "Do not require the subject to be a central narrative character. "
                    "Exclude props, setting details, weather, plants, and incidental "
                    "background entities.\n"
                )
            response = call_openai(
                scope_instruction
                + "Extract all applicable visual character subjects in this image prompt. "
                "Return a JSON array with subject_id, subject_phrase, name, aliases, "
                "features, and action for each subject. Always provide a useful name; "
                "if the prompt gives only a noun phrase, use that phrase as the name. "
                "subject_phrase should "
                "preserve the full noun phrase by which the character is referred to "
                "in this prompt, including "
                "pronouns and modifiers. Put visible identifying details in features. "
                "Return [] only when no identifiable character or animal is depicted "
                "or mentioned.\nPrompt: {}".format(prompt),
                temperature=0.2,
                max_output_tokens=600,
            )
            extracted = json.loads(response)
        except Exception as exc:
            print("[Controller] Subject extraction failed: {}".format(exc), flush=True)
            return []

        if isinstance(extracted, dict):
            extracted = extracted.get("subjects", [])
        if not isinstance(extracted, list):
            print("[Controller] Subject extraction returned invalid data.", flush=True)
            return []
        subjects = []
        for item in extracted:
            if not isinstance(item, dict):
                continue
            subject = dict(item)
            name = str(
                subject.get("name") or subject.get("subject_phrase") or ""
            ).strip()
            if not name:
                continue
            subject["name"] = name
            subject["subject_phrase"] = str(
                subject.get("subject_phrase") or name
            ).strip()
            subjects.append(subject)
        print(
            "[Controller] Identified {} subject(s).".format(len(subjects)), flush=True
        )
        return subjects

    def _new_subject_id(self, subject, reserved_ids):
        base = (
            str(subject.get("subject_id") or subject.get("name") or "subject")
            .strip()
            .casefold()
        )
        base = base or "subject"
        subject_id = base
        suffix = 2
        while subject_id in self.memory.long_term_memory or subject_id in reserved_ids:
            subject_id = "{}::{}".format(base, suffix)
            suffix += 1
        return subject_id

    def _resolve_subjects(self, prompt, subjects):
        identified = (
            self._identify_subjects(prompt, self.story_characters)
            if subjects is None
            else subjects
        )
        identified = [
            dict(subject)
            for subject in identified
            if str(subject.get("name") or "").strip()
        ]

        selected_memories = {}
        used_memory_ids = set()
        assigned_ids = set()
        for subject in identified:
            subject_name = str(subject.get("name") or "").strip()
            candidates = self.memory.match_subjects(
                subject_name,
                lexical_prompt="",
                include_below_threshold=True,
            )
            roster_subject_id = str(subject.get("subject_id") or "")
            roster_record = self.memory.long_term_memory.get(roster_subject_id)
            if roster_record is not None and roster_subject_id not in used_memory_ids:
                candidates = [
                    {
                        "subject_id": roster_subject_id,
                        "score": 1.0,
                        "subject": roster_record,
                    }
                ] + [
                    candidate
                    for candidate in candidates
                    if candidate["subject_id"] != roster_subject_id
                ]
            top_candidates = candidates[:3]
            candidate_summary = [
                "{}={:.3f}".format(item["subject_id"], item["score"])
                for item in top_candidates
            ]
            print(
                "[SubjectMatch] {!r} top candidates: {}".format(
                    subject_name, candidate_summary or "none (empty memory)"
                ),
                flush=True,
            )
            best_score = candidates[0]["score"] if candidates else None
            subject["best_memory_similarity"] = best_score
            subject["memory_similarity_threshold"] = self.memory.semantic_threshold
            match = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate["score"] >= self.memory.semantic_threshold
                    and candidate["subject_id"] not in used_memory_ids
                ),
                None,
            )
            if match:
                used_memory_ids.add(match["subject_id"])
                subject["subject_id"] = match["subject_id"]
                assigned_ids.add(match["subject_id"])
                subject["memory_match_score"] = match["score"]
                record = match["subject"]
                subject["features"] = record.get("features") or subject.get(
                    "features", ""
                )
                known_name = record.get("name")
                if known_name:
                    subject["name"] = known_name
                current_aliases = subject.get("aliases", [])
                if isinstance(current_aliases, str):
                    current_aliases = [current_aliases]
                elif not isinstance(current_aliases, (list, tuple)):
                    current_aliases = []
                subject["aliases"] = list(
                    dict.fromkeys(list(current_aliases) + record.get("aliases", []))
                )
                selected_memories[match["subject_id"]] = record
                print(
                    "[SubjectMatch] Decision: OLD subject {!r} matched {!r}; score={:.3f}, threshold={:.3f}.".format(
                        subject_name,
                        match["subject_id"],
                        match["score"],
                        self.memory.semantic_threshold,
                    ),
                    flush=True,
                )
            else:
                subject["subject_id"] = self._new_subject_id(
                    subject,
                    reserved_ids=assigned_ids,
                )
                assigned_ids.add(subject["subject_id"])
                score_text = (
                    "n/a" if best_score is None else "{:.3f}".format(best_score)
                )
                print(
                    "[SubjectMatch] Decision: NEW subject {!r}; best score={}, threshold={:.3f}.".format(
                        subject_name, score_text, self.memory.semantic_threshold
                    ),
                    flush=True,
                )

        context = {
            "short_term": list(self.memory.short_term_memory),
            "long_term": selected_memories,
        }
        known_ids = set(selected_memories)
        context["known_subjects"] = [
            subject for subject in identified if subject.get("subject_id") in known_ids
        ]
        context["new_subjects"] = [
            subject
            for subject in identified
            if subject.get("subject_id") not in known_ids
        ]
        return identified, context

    def _select_reference_image(self, context):
        # Combine distinct references when multiple remembered subjects appear.
        image_paths = []
        seen_paths = set()
        for subject in context["long_term"].values():
            image_path = subject.get("representative_image")
            if not image_path:
                continue
            path = Path(str(image_path)).expanduser()
            if not path.is_file():
                print(
                    "[Controller] Missing subject reference image: {}".format(path),
                    flush=True,
                )
                continue
            key = str(path.resolve())
            if key not in seen_paths:
                seen_paths.add(key)
                image_paths.append(path)

        if len(image_paths) == 1:
            return str(image_paths[0])
        if len(image_paths) > 1:
            try:
                from PIL import Image, ImageOps

                tile_size = (512, 512)
                gap = 16
                columns = 2
                rows = int(math.ceil(len(image_paths) / float(columns)))
                canvas = Image.new(
                    "RGB",
                    (
                        columns * tile_size[0] + (columns + 1) * gap,
                        rows * tile_size[1] + (rows + 1) * gap,
                    ),
                    color=(32, 32, 32),
                )
                for index, path in enumerate(image_paths):
                    with Image.open(path) as source:
                        tile = ImageOps.contain(source.convert("RGB"), tile_size)
                    x = gap + (index % columns) * (tile_size[0] + gap)
                    y = gap + (index // columns) * (tile_size[1] + gap)
                    canvas.paste(
                        tile,
                        (
                            x + (tile_size[0] - tile.width) // 2,
                            y + (tile_size[1] - tile.height) // 2,
                        ),
                    )
                cache_dir = (
                    Path(getattr(self.generator, "output_dir", Path.cwd()))
                    / "reference_images"
                )
                cache_dir.mkdir(parents=True, exist_ok=True)
                composite_path = cache_dir / "subjects_{}.png".format(uuid4().hex)
                canvas.save(composite_path)
                print(
                    "[Controller] Combined {} subject reference images: {}".format(
                        len(image_paths), composite_path
                    ),
                    flush=True,
                )
                return str(composite_path)
            except Exception as exc:
                print(
                    "[Controller] Could not combine subject references: {}".format(exc),
                    flush=True,
                )
                return str(image_paths[0])

        # New subjects start without a reference image.
        return None

    @staticmethod
    def _quality_score(report):
        scores = report.get("scores", {})
        if not scores:
            return float("-inf")

        values = []
        if "prompt_image" in scores:
            values.append(scores["prompt_image"])
        values.extend(scores.get("subjects", {}).values())
        action = scores.get("action_binding", {})
        if action:
            values.append(
                action.get("correct_binding", 0.0) - action.get("swapped_binding", 0.0)
            )
        return sum(values) / len(values) if values else float("-inf")

    @classmethod
    def _prefer_report(cls, candidate, current):
        if candidate.get("passed") != current.get("passed"):
            return candidate.get("passed", False)
        return cls._quality_score(candidate) > cls._quality_score(current)

    @staticmethod
    def _add_multi_subject_reconstruction_instruction(prompt, context):
        known_subjects = context["known_subjects"]
        if (
            len(known_subjects) < 2
            or "Treat the combined reference image only" in prompt
        ):
            return prompt
        return (
            "{} Treat the combined reference image only as a source for each "
            "character's identity and appearance. Create a newly composed, "
            "single coherent scene with all named characters together. Do not "
            "copy or edit the reference montage, its panel layout, borders, or "
            "source backgrounds. Integrate the characters naturally with "
            "consistent scale, perspective, lighting, and shadows; follow the "
            "requested poses, actions, and setting."
        ).format(prompt.strip())

    def process(
        self,
        prompt,
        turn,
        subjects=None,
        reference_image_override=None,
        published_filename=None,
        refine_prompt=True,
    ):
        print("\n[Controller] Starting turn {}.".format(turn), flush=True)
        identified_subjects, context = self._resolve_subjects(prompt, subjects)
        reference_image = (
            reference_image_override
            if reference_image_override is not None
            else self._select_reference_image(context)
        )
        print(
            "[Controller] Memory selection: {} known, {} new; reference image={}.".format(
                len(context["known_subjects"]),
                len(context["new_subjects"]),
                "yes" if reference_image else "no",
            ),
            flush=True,
        )

        if refine_prompt:
            print("[Controller] Refining prompt...", flush=True)
            refined_prompt = self.refiner.refine(prompt, context)
            if reference_image_override is None:
                refined_prompt = self._add_multi_subject_reconstruction_instruction(
                    refined_prompt, context
                )
            print(
                "[PromptRefiner] Refined prompt: {}".format(refined_prompt),
                flush=True,
            )
        else:
            # Keep edit instructions unchanged because they depend on the reference.
            refined_prompt = prompt.strip()
            print(
                "[PromptRefiner] Skipped for edit instruction; preserving original prompt.",
                flush=True,
            )
        print("[Controller] Generating first candidate...", flush=True)
        image_path = self.generator.generate(
            refined_prompt,
            reference_image,
            output_name="scene_{}_initial.png".format(turn),
        )
        print("[Controller] Verifying first candidate...", flush=True)
        report = self.verifier.verify(
            image_path,
            refined_prompt,
            subjects=identified_subjects,
            reference_image=reference_image,
        )

        initial_image_path = image_path
        initial_report = report
        retry_prompt = None
        retry_image = None
        retry_report = None
        if report["available"] and not report["passed"]:
            print(
                "[Controller] Verification failed; starting one targeted retry.",
                flush=True,
            )
            feedback = report.get("feedback", [])
            if refine_prompt:
                retry_prompt = self.refiner.refine(
                    refined_prompt,
                    context,
                    feedback=feedback,
                )
                if reference_image_override is None:
                    retry_prompt = self._add_multi_subject_reconstruction_instruction(
                        retry_prompt, context
                    )
            else:
                retry_prompt = prompt.strip()
            retry_label = (
                "[PromptRefiner] Retry refined prompt: {}"
                if refine_prompt
                else "[Controller] Retry prompt (original edit instruction): {}"
            )
            print(retry_label.format(retry_prompt), flush=True)
            retry_image = self.generator.generate(
                retry_prompt,
                reference_image,
                output_name="scene_{}_retry.png".format(turn),
            )
            print("[Controller] Verifying retry candidate...", flush=True)
            retry_report = self.verifier.verify(
                retry_image,
                retry_prompt,
                subjects=identified_subjects,
                reference_image=reference_image,
            )
            if retry_report["available"] and self._prefer_report(retry_report, report):
                image_path = retry_image
                report = retry_report

        if report["available"]:
            print(
                "[Controller] Final verification: passed={}, scores={}.".format(
                    report["passed"], report["scores"]
                ),
                flush=True,
            )
            if not report["passed"]:
                print(
                    "[Controller] Verification remained below threshold after retry.",
                    flush=True,
                )
        else:
            print(
                "[Controller] Verification unavailable: {}".format(report["error"]),
                flush=True,
            )

        image_path = self.generator.publish_scene(
            image_path, turn, filename=published_filename
        )
        self.last_run = {
            "turn": turn,
            "input_prompt": prompt,
            "refined_prompt": refined_prompt,
            "subjects": identified_subjects,
            "known_subjects": [
                subject.get("subject_id") for subject in context["known_subjects"]
            ],
            "new_subjects": [
                subject.get("subject_id") for subject in context["new_subjects"]
            ],
            "reference_image": reference_image,
            "initial_image": initial_image_path,
            "initial_verification": initial_report,
            "retry_prompt": retry_prompt,
            "retry_image": retry_image,
            "retry_verification": retry_report,
            "selected_image": image_path,
            "final_verification": report,
        }

        self.memory.record_turn(
            turn,
            prompt,
            image_path,
            subjects=identified_subjects,
        )
        return image_path
