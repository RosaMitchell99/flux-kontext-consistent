import re
from pathlib import Path

class Verifier:
    MODEL_ID = "openai/clip-vit-base-patch32"

    def __init__(
        self,
        model_id=MODEL_ID,
        prompt_threshold=0.22,
        subject_threshold=0.20,
        action_margin=0.0,
        model=None,
        processor=None,
    ):
        self.model_id = model_id
        self.prompt_threshold = prompt_threshold
        self.subject_threshold = subject_threshold
        self.action_margin = action_margin
        self.model = model
        self.processor = processor
        self._clip_load_error = None

    def _load_clip(self):
        if self.model is not None and self.processor is not None:
            return
        if self._clip_load_error is not None:
            raise RuntimeError(str(self._clip_load_error))
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            self._clip_load_error = RuntimeError(
                "CLIP verification requires torch and transformers."
            )
            raise self._clip_load_error from exc

        try:
            self.device = torch.device("cpu")
            print("[Verifier] Loading {} on CPU...".format(self.model_id), flush=True)
            self.model = (
                CLIPModel.from_pretrained(
                    self.model_id,
                )
                .to(self.device)
                .eval()
            )

            self.processor = CLIPProcessor.from_pretrained(
                self.model_id,
            )
            print("[Verifier] CLIP model ready.", flush=True)
        except Exception as exc:
            self._clip_load_error = exc
            raise

    @staticmethod
    def _open_image(image):
        if hasattr(image, "convert"):
            return image.convert("RGB")
        from PIL import Image

        return Image.open(Path(image)).convert("RGB")

    @staticmethod
    def _normalize(features):
        import torch.nn.functional as functional

        return functional.normalize(features, p=2, dim=-1)

    def _score_texts(self, image, texts):
        import torch

        self._load_clip()
        inputs = self.processor(
            images=image,
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with torch.inference_mode():
            outputs = self.model(**inputs)
            image_features = self._normalize(outputs.image_embeds)
            text_features = self._normalize(outputs.text_embeds)
            similarities = (text_features @ image_features.T).squeeze(-1)
        return [float(value) for value in similarities.cpu().tolist()]

    def subject_text_similarity(self, prompt, subject_record):
        query_name = str(prompt or "").strip()
        reference_name = str(subject_record.get("name") or "").strip()
        if not reference_name or not query_name:
            return 0.0
        normalize_name = lambda name: re.sub(
            r"^(?:a|an|the)\s+", "", name.casefold()
        ).strip()
        if normalize_name(query_name) == normalize_name(reference_name):
            return 1.0

        try:
            import torch

            self._load_clip()
            inputs = self.processor(
                text=[query_name, reference_name],
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            with torch.inference_mode():
                # Encode text directly; model.forward may require image inputs.
                features = self.model.get_text_features(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                )
                if hasattr(features, "pooler_output"):
                    features = features.pooler_output
                if features is None:
                    raise RuntimeError("CLIP returned no text features.")
                features = self._normalize(features)
                return float((features[0] @ features[1].T).item())
        except Exception as exc:
            print(
                "[Verifier] Subject similarity failed for {!r}: {}".format(
                    reference_name, exc
                ),
                flush=True,
            )
            return 0.0

    def verify(self, generated_image, prompt, subjects=None, reference_image=None):
        subjects = [item for item in (subjects or []) if isinstance(item, dict)]
        try:
            image = self._open_image(generated_image)
            texts = [prompt]
            subject_texts = []
            for subject in subjects:
                name = str(subject.get("name") or "").strip()
                features = str(subject.get("features") or "").strip()
                action = str(subject.get("action") or "").strip()
                parts = [part for part in (name, features, action) if part]
                if parts:
                    subject_texts.append((subject, " ".join(parts)))
                    texts.append(" ".join(parts))

            scores = self._score_texts(image, texts)
        except Exception as exc:
            print("[Verifier] Verification unavailable: {}".format(exc), flush=True)
            return {
                "available": False,
                "passed": True,
                "scores": {},
                "feedback": [],
                "error": str(exc),
            }

        prompt_score = scores[0]
        subject_scores = {}
        feedback = []
        passed = prompt_score >= self.prompt_threshold
        if not passed:
            feedback.append(
                "The image does not clearly match the full prompt. Preserve the requested scene and make its key visual details explicit."
            )

        for index, (subject, text) in enumerate(subject_texts, start=1):
            score = scores[index]
            name = str(subject.get("name") or "Subject")
            subject_scores[name] = score
            if score < self.subject_threshold:
                passed = False
                feedback.append(
                    "Make {} and its requested appearance/action more visually prominent: {}".format(
                        name, text
                    )
                )

        action_pairs = [
            (
                str(subject.get("name") or "").strip(),
                str(subject.get("action") or "").strip(),
            )
            for subject in subjects
            if str(subject.get("name", "")).strip()
            and str(subject.get("action", "")).strip()
        ]
        action_scores = {}
        if len(action_pairs) > 1:
            correct_text = "; ".join(
                "{}: {}".format(name, action) for name, action in action_pairs
            )
            swapped_text = "; ".join(
                "{}: {}".format(name, action_pairs[(index + 1) % len(action_pairs)][1])
                for index, (name, _) in enumerate(action_pairs)
            )
            correct_score, swapped_score = self._score_texts(
                image, [correct_text, swapped_text]
            )
            action_scores = {
                "correct_binding": correct_score,
                "swapped_binding": swapped_score,
            }
            if swapped_score >= correct_score + self.action_margin:
                passed = False
                feedback.append(
                    "Strengthen the subject-to-action assignments: {}".format(
                        correct_text
                    )
                )

        reference_score = None
        if reference_image is not None:
            try:
                reference = self._open_image(reference_image)
                self._load_clip()
                import torch

                image_inputs = self.processor(
                    images=[image, reference], return_tensors="pt"
                )
                with torch.inference_mode():
                    image_features = self._normalize(
                        self.model(
                            pixel_values=image_inputs["pixel_values"]
                        ).image_embeds
                    )
                    reference_score = float(
                        (image_features[0] @ image_features[1].T).item()
                    )
            except Exception:
                reference_score = None

        print(
            "[Verifier] Scores: prompt={:.3f}, subjects={}, passed={}.".format(
                prompt_score,
                {name: round(score, 3) for name, score in subject_scores.items()},
                passed,
            ),
            flush=True,
        )
        return {
            "available": True,
            "passed": passed,
            "scores": {
                "prompt_image": prompt_score,
                "subjects": subject_scores,
                "action_binding": action_scores,
                "reference_image": reference_score,
            },
            "feedback": feedback,
        }
