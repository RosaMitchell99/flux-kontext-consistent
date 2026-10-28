import json
import re
import sys
import uuid
from pathlib import Path

import streamlit as st

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agent.controller import AgentController
from agent.generator import Generator


def _contextual_subjects(prompt, controller):
    records = controller.memory.long_term_memory
    roster = [
        {
            "subject_id": subject_id,
            "name": record.get("name", ""),
            "aliases": record.get("aliases", []),
            "features": record.get("features", ""),
            "first_appearance_prompt": record.get("representative_prompt", ""),
        }
        for subject_id, record in records.items()
    ]
    recent_turns = [
        {"turn": item.get("turn"), "prompt": item.get("prompt", "")}
        for item in controller.memory.short_term_memory
    ]

    if roster:
        try:
            from agent.llm_utils import call_openai

            response = call_openai(
                "Identify only the central character(s) in the user's image prompt. "
                "Use the remembered subjects below to resolve names, aliases, and "
                "natural-language references such as pronouns or descriptions. If a "
                "remembered subject is referenced, return its exact subject_id and "
                "canonical name; subject_phrase should preserve the expression used "
                "in the current prompt. Return a new subject with subject_id null if "
                "it is central but is not in memory. Exclude setting details, props, "
                "weather, and incidental entities. Return JSON only, as an array of "
                "objects with subject_id, subject_phrase, name, aliases, features, "
                "and action. Return [] if there is no central character.\n"
                "Remembered subjects: {}\nRecent conversation: {}\nPrompt: {}".format(
                    json.dumps(roster, ensure_ascii=False),
                    json.dumps(recent_turns, ensure_ascii=False),
                    prompt,
                ),
                temperature=0.1,
                max_output_tokens=400,
                system_prompt="You resolve character references for an image generation conversation. Return valid JSON only.",
            )
            extracted = json.loads(response)
            if isinstance(extracted, dict):
                extracted = extracted.get("subjects", [])
            if isinstance(extracted, list):
                result = []
                for item in extracted:
                    if (
                        not isinstance(item, dict)
                        or not str(item.get("name") or "").strip()
                    ):
                        continue
                    subject_id = item.get("subject_id")
                    if subject_id is not None and str(subject_id) in records:
                        subject_id = str(subject_id)
                        remembered = records[subject_id]
                        result.append(
                            {
                                "subject_id": subject_id,
                                "subject_phrase": str(
                                    item.get("subject_phrase") or remembered.get("name")
                                ),
                                "name": remembered.get("name"),
                                "aliases": remembered.get("aliases", []),
                                "features": remembered.get("features", ""),
                                "action": str(item.get("action") or ""),
                            }
                        )
                    else:
                        result.append(
                            {
                                "subject_phrase": str(
                                    item.get("subject_phrase") or item["name"]
                                ),
                                "name": str(item["name"]).strip(),
                                "aliases": item.get("aliases", []),
                                "features": str(item.get("features") or ""),
                                "action": str(item.get("action") or ""),
                            }
                        )
                return result, "contextual_llm"

        except Exception as exc:
            print(
                "[UI] Contextual subject resolution unavailable: {}".format(exc),
                flush=True,
            )

    return _fallback_subjects(prompt, controller), "name_or_single_subject_fallback"


def _fallback_subjects(prompt, controller):
    records = controller.memory.long_term_memory
    lowered = prompt.casefold()
    referenced = []
    for subject_id, record in records.items():
        terms = [record.get("name", "")] + list(record.get("aliases", []))
        if any(
            term
            and re.search(r"(?<!\w){}(?!\w)".format(re.escape(str(term))), prompt, re.I)
            for term in terms
        ):
            referenced.append(
                {
                    "subject_id": subject_id,
                    "subject_phrase": record.get("name", ""),
                    "name": record.get("name", ""),
                    "aliases": record.get("aliases", []),
                    "features": record.get("features", ""),
                }
            )

    pronoun = re.search(
        r"\b(?:it|its|he|him|his|she|her|hers|they|them|their|this character|that character|the subject)\b",
        lowered,
    )
    if not referenced and len(records) == 1 and pronoun:
        subject_id, record = next(iter(records.items()))
        referenced.append(
            {
                "subject_id": subject_id,
                "subject_phrase": pronoun.group(0),
                "name": record.get("name", ""),
                "aliases": record.get("aliases", []),
                "features": record.get("features", ""),
            }
        )

    try:
        extracted = AgentController._identify_subjects(prompt)
    except Exception:
        extracted = []
    seen = {str(item.get("subject_id", "")).casefold() for item in referenced}
    known_names = {
        str(records[item["subject_id"]].get("name", "")).casefold()
        for item in referenced
        if item.get("subject_id") in records
    }
    for subject in extracted:
        name = str(subject.get("name") or "").strip()
        if not name or name.casefold() in known_names:
            continue
        subject_id = str(subject.get("subject_id") or "").casefold()
        if subject_id and subject_id in seen:
            continue
        referenced.append(subject)
        if subject_id:
            seen.add(subject_id)
    return referenced


def _new_controller():
    session_id = uuid.uuid4().hex[:12]
    output_dir = SRC_DIR / "agent" / "outputs" / "ui_sessions" / session_id
    return AgentController(generator=Generator(output_dir=output_dir)), session_id


def _initialize_state():
    if "controller" not in st.session_state:
        controller, session_id = _new_controller()
        st.session_state.controller = controller
        st.session_state.session_id = session_id
        st.session_state.messages = []
        st.session_state.next_turn = 0


def _render_message(message):
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
            return
        if message.get("error"):
            st.error(message["error"])
            return
        image_path = message.get("image_path")
        if image_path and Path(image_path).is_file():
            st.image(image_path, use_container_width=True)
        st.markdown("**Refined prompt**")
        st.write(message.get("refined_prompt", ""))
        if message.get("referenced_subjects"):
            st.caption(
                "Recalled subjects: {}".format(
                    ", ".join(message["referenced_subjects"])
                )
            )
        with st.expander("Generation details"):
            st.json(
                {
                    "subjects": message.get("subjects", []),
                    "known_subject_ids": message.get("known_subject_ids", []),
                    "new_subject_ids": message.get("new_subject_ids", []),
                    "reference_image": message.get("reference_image"),
                    "verification": message.get("verification", {}),
                    "subject_resolution": message.get("subject_resolution"),
                }
            )


def _render_memory(controller):
    with st.sidebar:
        st.subheader("Remembered subjects")
        records = controller.memory.long_term_memory
        if not records:
            st.caption("Subjects from this session will appear here.")
        for subject_id, record in records.items():
            with st.expander(
                "{} · turn {}".format(
                    record.get("name", subject_id), record.get("first_turn", "?")
                )
            ):
                if (
                    record.get("representative_image")
                    and Path(record["representative_image"]).is_file()
                ):
                    st.image(record["representative_image"], use_container_width=True)
                st.caption("Memory ID: {}".format(subject_id))
                if record.get("features"):
                    st.write(record["features"])
        st.divider()
        st.caption("Session: {}".format(st.session_state.session_id))
        if st.button("Start a new session", use_container_width=True):
            for key in ("controller", "session_id", "messages", "next_turn"):
                st.session_state.pop(key, None)
            st.rerun()


def main():
    st.set_page_config(page_title="FLUX Story Agent", page_icon="🎨", layout="wide")
    _initialize_state()
    controller = st.session_state.controller

    st.title("FLUX Story Agent")
    st.caption(
        "Describe a scene or edit in natural language. Characters mentioned in "
        "earlier turns can be recalled by name, alias, or contextual reference."
    )

    _render_memory(controller)

    mode = st.radio(
        "Interaction mode",
        ("Continue story", "Edit previous image"),
        horizontal=True,
        disabled=not bool(controller.memory.short_term_memory),
        help="Edit mode uses the most recent generated image as the FLUX Kontext reference.",
    )
    for message in st.session_state.messages:
        _render_message(message)

    prompt = st.chat_input("Describe the next scene or edit…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    turn = st.session_state.next_turn
    reference_override = None
    if mode == "Edit previous image" and controller.memory.short_term_memory:
        reference_override = controller.memory.short_term_memory[-1].get("image_path")

    with st.chat_message("assistant"):
        with st.spinner("Resolving remembered subjects and generating the image…"):
            subjects, resolution_method = _contextual_subjects(prompt, controller)
            try:
                image_path = controller.process(
                    prompt,
                    turn,
                    subjects=subjects,
                    reference_image_override=reference_override,
                )
                debug = controller.last_run or {}
                record = {
                    "role": "assistant",
                    "image_path": image_path,
                    "refined_prompt": debug.get("refined_prompt", ""),
                    "subjects": debug.get("subjects", []),
                    "known_subject_ids": debug.get("known_subjects", []),
                    "new_subject_ids": debug.get("new_subjects", []),
                    "referenced_subjects": [
                        str(subject.get("name"))
                        for subject in debug.get("subjects", [])
                        if subject.get("subject_id") in debug.get("known_subjects", [])
                    ],
                    "reference_image": debug.get("reference_image"),
                    "verification": debug.get("final_verification", {}),
                    "subject_resolution": resolution_method,
                }
                st.session_state.messages.append(record)
                st.session_state.next_turn += 1
            except Exception as exc:
                st.session_state.messages.append(
                    {"role": "assistant", "error": str(exc)}
                )
            st.rerun()


if __name__ == "__main__":
    main()
