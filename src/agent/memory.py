import json
import re
from pathlib import Path


class MemorySystem:

    _GENERIC_REFERENCES = {
        "he",
        "she",
        "they",
        "it",
        "him",
        "her",
        "them",
        "his",
        "their",
        "this person",
        "that person",
        "the person",
        "this character",
        "that character",
        "the character",
        "someone",
        "somebody",
    }

    def __init__(
        self, short_term_size=1, semantic_matcher=None, semantic_threshold=0.25
    ):
        if short_term_size < 0:
            raise ValueError("short_term_size must be non-negative")
        self.short_term_size = short_term_size
        self.short_term_memory = []
        self.long_term_memory = {}
        self.semantic_matcher = semantic_matcher
        self.semantic_threshold = semantic_threshold

    def clear_short_term(self):
        self.short_term_memory.clear()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "short_term_size": self.short_term_size,
            "semantic_threshold": self.semantic_threshold,
            "short_term_memory": self.short_term_memory,
            "long_term_memory": self.long_term_memory,
        }
        temporary_path = path.with_name(path.name + ".tmp")
        with temporary_path.open("w", encoding="utf-8") as destination:
            json.dump(payload, destination, indent=2, ensure_ascii=False)
        temporary_path.replace(path)

    @classmethod
    def load(cls, path, semantic_matcher=None):
        path = Path(path)
        if not path.is_file():
            return cls(semantic_matcher=semantic_matcher)
        with path.open("r", encoding="utf-8") as source:
            payload = json.load(source)
        memory = cls(
            short_term_size=int(payload.get("short_term_size", 1)),
            semantic_matcher=semantic_matcher,
            semantic_threshold=float(payload.get("semantic_threshold", 0.25)),
        )
        short_term = payload.get("short_term_memory", [])
        long_term = payload.get("long_term_memory", {})
        memory.short_term_memory = short_term if isinstance(short_term, list) else []
        memory.long_term_memory = long_term if isinstance(long_term, dict) else {}
        return memory

    def add_short_term(self, turn, prompt, image_path):
        if self.short_term_size == 0:
            return
        self.short_term_memory.append(
            {"turn": turn, "prompt": prompt, "image_path": image_path}
        )
        overflow = len(self.short_term_memory) - self.short_term_size
        if overflow > 0:
            del self.short_term_memory[:overflow]

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        return [str(item) for item in value if item is not None and str(item).strip()]

    @staticmethod
    def _append_unique(items, value):
        if value and value not in items:
            items.append(value)

    def add_long_term(
        self,
        subject_id,
        name,
        features="",
        image_path=None,
        description="",
        aliases=None,
        prompt=None,
        turn=None,
    ):
        if not name or not str(name).strip():
            raise ValueError("name must be a non-empty subject name")
        name = str(name).strip()
        subject_id = str(subject_id or name.casefold()).strip()
        if not subject_id:
            raise ValueError("subject_id must be a non-empty string")

        if subject_id in self.long_term_memory:
            return subject_id

        feature_texts = self._as_list(features)
        description_texts = self._as_list(description)
        stored_aliases = []
        for alias in self._as_list(aliases):
            self._append_unique(stored_aliases, alias.strip())

        self.long_term_memory[subject_id] = {
            "name": name,
            "aliases": stored_aliases,
            "features": "; ".join(text.strip() for text in feature_texts),
            "description": "; ".join(text.strip() for text in description_texts),
            "representative_prompt": prompt,
            "representative_image": image_path,
            "first_turn": turn,
        }
        return subject_id

    def record_turn(self, turn, prompt, image_path, subjects=None):
        existing_ids = set(self.long_term_memory)
        self.add_short_term(turn, prompt, image_path)
        for subject in subjects or []:
            if not isinstance(subject, dict) or not subject.get("name"):
                continue
            self.add_long_term(
                subject_id=subject.get("subject_id", subject.get("id")),
                name=subject.get("name"),
                aliases=subject.get("aliases"),
                features=subject.get("features", ""),
                description=subject.get("description", ""),
                prompt=prompt,
                image_path=image_path,
                turn=turn,
            )
        added_ids = sorted(set(self.long_term_memory) - existing_ids)
        retained_ids = sorted(set(self.long_term_memory).intersection(existing_ids))
        print(
            "[Memory] Turn {}: short-term prompt/image stored ({}/{} turns); "
            "new representative records={}; existing first-appearance records retained={}.".format(
                turn,
                len(self.short_term_memory),
                self.short_term_size,
                added_ids or "none",
                retained_ids or "none",
            ),
            flush=True,
        )

    @staticmethod
    def _contains_phrase(prompt, phrase):
        phrase = str(phrase).strip()
        if not phrase:
            return False
        pattern = r"(?<!\w){}(?!\w)".format(re.escape(phrase))
        return re.search(pattern, prompt, flags=re.IGNORECASE) is not None

    def match_subjects(
        self, prompt, lexical_prompt=None, include_below_threshold=False
    ):
        if not isinstance(prompt, str) or not prompt.strip():
            return []
        lexical_text = prompt if lexical_prompt is None else lexical_prompt

        matches = []
        for subject_id, record in self.long_term_memory.items():
            terms = [record.get("name", "")] + record.get("aliases", [])
            lexical_hits = [
                term
                for term in terms
                if str(term).strip().casefold() not in self._GENERIC_REFERENCES
                and self._contains_phrase(lexical_text, term)
            ]
            score = 1.0 if lexical_hits else 0.0
            match_type = "name_or_alias" if lexical_hits else None

            if self.semantic_matcher is not None:
                try:
                    semantic_score = float(self.semantic_matcher(prompt, record))
                except (TypeError, ValueError):
                    semantic_score = 0.0
                if semantic_score > score:
                    score = semantic_score
                    match_type = "semantic"

            if (
                include_below_threshold
                or lexical_hits
                or score >= self.semantic_threshold
            ):
                matches.append(
                    {
                        "subject_id": subject_id,
                        "score": score,
                        "match_type": match_type,
                        "matched_terms": lexical_hits,
                        "subject": record,
                    }
                )

        matches.sort(key=lambda item: item["score"], reverse=True)
        return matches

    def best_match(self, prompt, lexical_prompt=None, threshold=None):
        matches = self.match_subjects(prompt, lexical_prompt=lexical_prompt)
        if not matches:
            return None
        best = max(matches, key=lambda item: item["score"])
        minimum = self.semantic_threshold if threshold is None else threshold
        return best if best["score"] >= minimum else None

    def get_context(self, prompt=None):
        if prompt is None:
            long_term = self.long_term_memory
        else:
            long_term = {
                match["subject_id"]: match["subject"]
                for match in self.match_subjects(prompt)
            }
        return {
            "short_term": list(self.short_term_memory),
            "long_term": long_term,
        }

    def find_subject(self, prompt):
        matches = self.match_subjects(prompt)
        return matches[0]["subject_id"] if matches else None
