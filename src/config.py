import os
import re
from typing import Dict, List

from loguru import logger
from openai import OpenAI
from tqdm import tqdm

logger.remove()

LOG_FORMAT = (
    "<fg #9f9f9f>{time:hh:mm:ss A}</fg #9f9f9f> "
    "<level>{level: <8}</level>"
    " <fg #9f9f9f>|</fg #9f9f9f> "
    "<level>{message}</level>"
)

logger.level("DEBUG", color="<fg #9f9f9f>", icon=" ")
logger.level("INFO", color="<fg #f6f6f6>", icon=" ")
logger.level("SUCCESS", color="<fg #b4fa72>", icon=" ")
logger.level("WARNING", color="<fg #fbbf24>", icon=" ")
logger.level("ERROR", color="<fg #f7768e>", icon=" ")

logger.add(
    lambda msg: tqdm.write(msg, end=""),
    format=LOG_FORMAT,
    level="DEBUG",
    colorize=True,
)

# "/media/milab-7/TS_dedup"
DEFAULT_DATASET_PATH = "/Users/mohammadabdulmannansami/Documents/CM/data"

PERSONAS: Dict[str, Dict] = {
    "beginner": {
        "description": "A developer new to TypeScript, still learning the basics of types and interfaces.",
        "persona": "beginner",
        "prompt_style": "Inexperienced and sometimes confused by error messages. Often unsure what the error means. Example: I wrote this code but it's showing some red squiggly lines and I don't understand what it wants me to do.",
        "traits": [
            "Asks basic questions about TypeScript concepts",
            "May not understand advanced type system features",
            "Appreciates detailed explanations",
            "Sometimes asks for clarification",
            "May express confusion about error messages",
            "May misidentify the error or describe symptoms incorrectly",
            "Sometimes tries a wrong fix and asks why it didn't work",
            "Uses informal language and may describe types imprecisely",
        ],
    },
    "intermediate": {
        "description": "A developer with some experience in TypeScript, comfortable with common patterns but not an expert.",
        "persona": "intermediate",
        "prompt_style": "Some experience, can describe errors with some detail. Given a buggy code, example: I am seeing type errors in this code, can you help me understand and fix them?",
        "traits": [
            "Understands basic concepts but struggles with advanced types",
            "Asks about best practices",
            "May ask follow-up questions about edge cases",
            "Interested in why something works a certain way",
            "Familiar with common patterns",
        ],
    },
    "advanced": {
        "description": "An advanced TypeScript developer with deep understanding of the type system and best practices.",
        "persona": "advanced",
        "prompt_style": "Expert level, can provide detailed explanations and solutions. Given a buggy code, example: I am encountering complex type errors in this code, can you help me debug and optimize it?",
        "traits": [
            "Asks precise, technical questions",
            "May inquire about type inference details",
            "Interested in performance implications",
            "May ask about alternative approaches",
            "Understands advanced type system features",
        ],
    },
}


def create_client(role: str = "assistant") -> OpenAI:
    """
    Create an OpenAI-compatible client for the given role.

    Args:
        role: "assistant" for the main model, "user" for the user-simulation fallback (Qwen).

    Returns:
        An OpenAI client instance.
    """
    if role == "user":
        api_key = os.getenv("USER_API_KEY", "not-needed")
        base_url = os.getenv("USER_BASE_URL", "http://0.0.0.0:8000/v1")
    else:
        api_key = os.getenv("ASSISTANT_API_KEY", os.getenv("API_KEY"))
        base_url = os.getenv("ASSISTANT_BASE_URL", os.getenv("BASE_URL"))

    return OpenAI(api_key=api_key, base_url=base_url)


def extract_json_from_response(content: str) -> str:
    """
    Extract JSON from an LLM response that may be wrapped in markdown code blocks.

    Handles:
      - ```json ... ```
      - ``` ... ```
      - <think>...</think> reasoning tags (Kimi / Qwen models)
      - Raw JSON with surrounding prose
    """
    if not content or not content.strip():
        return content

    # Strip <think>...</think> tags first (Kimi K2.5, Qwen3, etc.)
    content = re.sub(
        r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE
    ).strip()

    match = re.search(r"```(?:json)?\s*\n?(.*?)```", content, re.DOTALL | re.IGNORECASE)
    if match and match.group(1).strip():
        return match.group(1).strip()

    # Fallback: extract the first top-level JSON object from the response.
    brace_start = content.find("{")
    brace_end = content.rfind("}")
    if brace_start != -1 and brace_end > brace_start:
        return content[brace_start : brace_end + 1]

    return content


def format_conversation_for_prompt(
    messages: List[Dict[str, str]], max_chars: int = 300
) -> str:
    """Format conversation history for including in prompts."""
    formatted = []
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"][:max_chars]
        formatted.append(f"{role}: {content}")
    return "\n\n".join(formatted)
