import os
from pathlib import Path
from typing import Any, cast
import httpx
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set.")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
    http_client=cast(Any, httpx.Client()),
)

from openai import OpenAI, OpenAIError


def call_openai(
    prompt,
    temperature=0.8,
    max_output_tokens=2048,
    system_prompt="You are a helpful assistant.",
):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_completion_tokens=max_output_tokens,
        )

        content = response.choices[0].message.content or ""
        return content.strip().replace("```json", "").replace("```", "")

    except OpenAIError as e:
        print(f"OpenAI API error: {type(e).__name__}: {e}")
        raise
