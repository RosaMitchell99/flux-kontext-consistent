import re


class PromptRefiner:

    _WEAK_VISUAL_PHRASES = (
        (
            r"\blisten(?:s|ed|ing)?\s+to\s+(?:the\s+)?music\b",
            "wearing headphones and moving lightly to the beat",
        ),
        (
            r"\blisten(?:s|ed|ing)?\b",
            "turning toward the sound with an attentive expression",
        ),
        (r"\bhear(?:s|d|ing)?\b", "turning toward the sound with an alert expression"),
        (r"\bwatch(?:es|ed|ing)?\b", "keeping a steady gaze on the subject"),
        (r"\bsee(?:s|ing)?\b", "looking directly at the subject"),
        (r"\blook(?:s|ed|ing)?\b", "gazing toward the subject"),
        (r"\bthink(?:s|ing)?\b", "pausing with a thoughtful expression"),
        (r"\bconsider(?:s|ed|ing)?\b", "pausing to examine the choice"),
        (r"\bwonder(?:s|ed|ing)?\b", "showing a curious, slightly puzzled expression"),
        (
            r"\bimagin(?:e|es|ed|ing)\b",
            "picturing the scene with a distant, dreamy gaze",
        ),
        (r"\bremember(?:s|ed|ing)?\b", "looking reflective and lost in thought"),
        (r"\bnotice(?:s|d|ing)?\b", "turning attention toward the detail"),
        (r"\bobserve(?:s|d|ing)?\b", "studying the scene closely"),
        (
            r"\bfeel(?:s|ing|t)?\b",
            "showing the emotion through facial expression and posture",
        ),
        (r"\bhope(?:s|d|ing)?\b", "looking expectant and hopeful"),
        (r"\bwait(?:s|ed|ing)?\b", "holding an attentive waiting posture"),
    )

    def __init__(self, max_prompt_chars=1200, llm_enabled=True):
        self.max_prompt_chars = max_prompt_chars
        self.llm_enabled = llm_enabled

    def _apply_pattern_matching(self, prompt):
        refined = prompt
        for pattern, replacement in self._WEAK_VISUAL_PHRASES:
            regex = re.compile(pattern, re.IGNORECASE)
            refined = regex.sub(replacement, refined)
        return refined

    def _rule_based_refinement(self, prompt, context):
        refined = self._apply_pattern_matching(prompt).strip()
        if not refined:
            return refined

        long_term = context.get("long_term", {}) if isinstance(context, dict) else {}
        if isinstance(long_term, dict):
            additions = []
            lowered = refined.casefold()
            known_subjects = context.get("known_subjects", [])
            if "known_subjects" in context:
                # Prefer subject identities resolved by the controller.
                remembered = []
                for current_subject in known_subjects:
                    if not isinstance(current_subject, dict):
                        continue
                    subject_id = current_subject.get("subject_id")
                    record = long_term.get(subject_id)
                    if isinstance(record, dict):
                        remembered.append((current_subject, record))
            else:
                # Support contexts that only include long-term records.
                remembered = []
                for record in long_term.values():
                    if not isinstance(record, dict):
                        continue
                    aliases = record.get("aliases", [])
                    if isinstance(aliases, str):
                        aliases = [aliases]
                    identity_terms = [record.get("name", "")] + aliases
                    if any(
                        term and str(term).casefold() in lowered
                        for term in identity_terms
                    ):
                        remembered.append(({"name": record.get("name", "")}, record))

            for current_subject, record in remembered:
                features = str(record.get("features", "")).strip()
                if not features or features.casefold() in lowered:
                    continue
                reference = str(
                    current_subject.get("subject_phrase")
                    or current_subject.get("name")
                    or record.get("name")
                    or "the subject"
                ).strip()
                additions.append(
                    "Preserve the remembered appearance of {}: {}".format(
                        reference, features
                    )
                )
            if additions:
                memory_suffix = "; ".join(additions)
                separator = "; "
                if self.max_prompt_chars is not None:
                    prefix_limit = max(
                        0,
                        self.max_prompt_chars - len(separator) - len(memory_suffix),
                    )
                    if len(refined) > prefix_limit:
                        if prefix_limit == 0:
                            refined = ""
                        else:
                            refined = (
                                refined[:prefix_limit].rsplit(" ", 1)[0].rstrip(" ,;")
                            )
                refined = "{}{}{}".format(
                    refined, separator if refined else "", memory_suffix
                )

        return self._truncate(refined)

    def _truncate(self, prompt):
        if self.max_prompt_chars is None or len(prompt) <= self.max_prompt_chars:
            return prompt
        shortened = prompt[: self.max_prompt_chars].rsplit(" ", 1)[0].rstrip(" ,;")
        return shortened or prompt[: self.max_prompt_chars]

    def refine(self, prompt, context=None, feedback=None):
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        context = context or {}
        rule_based = self._rule_based_refinement(prompt, context)
        if rule_based != prompt.strip():
            print("[PromptRefiner] Rule and memory refinement applied.", flush=True)
        else:
            print("[PromptRefiner] No weak-verb or memory edits needed.", flush=True)
        if not rule_based or not self.llm_enabled:
            if not self.llm_enabled:
                print(
                    "[PromptRefiner] LLM disabled; using rule-based prompt.", flush=True
                )
            return rule_based

        try:
            from .llm_utils import call_openai

            camera_view = call_openai(
                "Choose one camera view for this scene: close-up, medium shot, "
                "wide shot, over-the-shoulder, eye-level, high-angle, low-angle, "
                "or bird's-eye view. Return only the view.\nScene: {}".format(
                    rule_based
                ),
                temperature=0.2,
                max_output_tokens=30,
            )
        except Exception as exc:
            camera_view = ""
            print(
                "[PromptRefiner] Camera selection unavailable: {}".format(exc),
                flush=True,
            )

        if isinstance(camera_view, str):
            camera_view = camera_view.strip().lower().strip(" .,:;\"'`")
            allowed_views = (
                "close-up",
                "medium shot",
                "wide shot",
                "over-the-shoulder",
                "eye-level",
                "high-angle",
                "low-angle",
                "bird's-eye view",
            )
            camera_view = next(
                (view for view in allowed_views if camera_view == view), ""
            )
        else:
            camera_view = ""

        prompt_with_camera = rule_based
        if camera_view:
            prompt_with_camera = "{}, camera view: {}".format(rule_based, camera_view)
            print(
                "[PromptRefiner] Selected camera view: {}.".format(camera_view),
                flush=True,
            )
        else:
            print("[PromptRefiner] No camera view selected.", flush=True)

        feedback_text = ""
        if feedback:
            if isinstance(feedback, (list, tuple)):
                feedback = "; ".join(str(item) for item in feedback if item)
            feedback_text = " Verification feedback: {}.".format(feedback)

        try:
            normalized = call_openai(
                "Format this image prompt as Subject 1: appearance and action; "
                "Subject 2: appearance and action; Scene: setting and composition. "
                "Keep the remembered appearance details for each identified subject. "
                "Keep only relevant fields, preserve the meaning, apply any verification "
                "feedback, and return only the formatted prompt.{}\nPrompt: {}".format(
                    feedback_text, prompt_with_camera
                ),
                temperature=0.2,
                max_output_tokens=350,
            )
            if isinstance(normalized, str) and normalized.strip():
                print("[PromptRefiner] Prompt normalization completed.", flush=True)
                return self._truncate(normalized.strip())
            print(
                "[PromptRefiner] Normalization returned no text; using fallback.",
                flush=True,
            )
        except Exception as exc:
            print(
                "[PromptRefiner] Normalization unavailable: {}".format(exc), flush=True
            )
        return self._truncate(prompt_with_camera)
