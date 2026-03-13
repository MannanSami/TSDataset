import asyncio
import json
from typing import Optional

from config import OPENCODE_MODEL, OPENCODE_PATH, logger


class OpenCodeServer:
    """Manages the opencode serve lifecycle."""

    def __init__(self, port: int = 14000):
        self.port = port
        self._process: Optional[asyncio.subprocess.Process] = None

    async def start(self) -> None:
        """Spawn `opencode serve --port <port>` as a background subprocess."""
        self._process = await asyncio.create_subprocess_exec(
            OPENCODE_PATH, "serve", "--port", str(self.port),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        # Give the server a moment to bind
        await asyncio.sleep(2)
        if self._process.returncode is not None:
            raise RuntimeError(
                f"OpenCode server exited immediately with code {self._process.returncode}"
            )
        logger.info(f"OpenCode server started on port {self.port} (pid {self._process.pid})")

    async def stop(self) -> None:
        """Terminate the server subprocess."""
        if self._process and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._process.kill()
            logger.info("OpenCode server stopped")

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


def build_combined_prompt(system_prompt: str, user_prompt: str) -> str:
    """Combine system and user prompts into a single prompt for opencode."""
    if not system_prompt or not system_prompt.strip():
        return user_prompt
    return f"<INSTRUCTIONS>\n{system_prompt}\n</INSTRUCTIONS>\n\n{user_prompt}"


def parse_opencode_output(stdout: str) -> str:
    """Parse JSON-lines output from opencode, collecting text events."""
    parts = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
            if event.get("type") == "text":
                part = event.get("part", {})
                if isinstance(part, dict):
                    text = part.get("text", "")
                elif isinstance(part, str):
                    text = part
                else:
                    text = ""
                if text:
                    parts.append(text)
        except json.JSONDecodeError:
            # Non-JSON line — might be raw text output
            if line and not line.startswith("{"):
                parts.append(line)
    # If no structured events were parsed, return raw stdout
    if not parts:
        return stdout.strip()
    return "".join(parts)


async def call_opencode(
    system_prompt: str,
    user_prompt: str,
    attach_url: str,
    model: str = OPENCODE_MODEL,
    timeout: int = 180,
    max_retries: int = 3,
) -> str:
    """
    Call opencode via `opencode run` attached to a running server.

    Retries with exponential backoff on failure.
    """
    combined = build_combined_prompt(system_prompt, user_prompt)

    for attempt in range(max_retries):
        try:
            proc = await asyncio.create_subprocess_exec(
                OPENCODE_PATH, "run", combined,
                "--format", "json",
                "--attach", attach_url,
                "-m", model,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
            stdout_str = stdout_bytes.decode("utf-8", errors="replace")

            if proc.returncode != 0:
                stderr_str = stderr_bytes.decode("utf-8", errors="replace")
                raise RuntimeError(
                    f"opencode exited with code {proc.returncode}: {stderr_str[:300]}"
                )

            result = parse_opencode_output(stdout_str)
            if result:
                return result
            raise RuntimeError("opencode returned empty output")

        except asyncio.TimeoutError:
            logger.warning(
                f"opencode call timed out (attempt {attempt + 1}/{max_retries})"
            )
            # Kill the timed-out subprocess to avoid zombies
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
        except Exception as e:
            logger.warning(
                f"opencode call failed (attempt {attempt + 1}/{max_retries}): {e}"
            )

        if attempt < max_retries - 1:
            wait = 2 ** (attempt + 1)
            logger.debug(f"Retrying in {wait}s...")
            await asyncio.sleep(wait)

    raise RuntimeError(f"opencode call failed after {max_retries} retries")


def serialize_conversation(
    system_prompt: str, messages: list[dict]
) -> tuple[str, str]:
    """
    Serialize a multi-turn conversation into a (system, user) prompt pair.

    Packs the message history into the user prompt so that opencode
    (which only accepts a single prompt string) can see the full context.
    """
    history_lines = []
    for msg in messages:
        role = msg["role"].upper()
        content = msg.get("content", "")
        history_lines.append(f"[{role}]\n{content}")

    user_prompt = (
        "Here is the conversation so far:\n\n"
        + "\n\n".join(history_lines)
        + "\n\nPlease provide your next response as the ASSISTANT."
    )
    return system_prompt, user_prompt
