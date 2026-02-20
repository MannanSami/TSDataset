import os
import random
import re
from typing import Dict, Optional

from openai.types.chat import ChatCompletionMessageParam

from config import PERSONAS, create_client, logger
from user_simulator import UserSimulator


class ConversationGenerator:
    """Generates debugging conversations in OpenAI format with synthetic user interactions."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize the conversation generator with assistant model."""
        resolved_model = model or os.getenv("ASSISTANT_MODEL")
        if not resolved_model:
            raise ValueError(
                "No model specified. Set ASSISTANT_MODEL env var or pass model parameter."
            )
        self.model: str = resolved_model

        try:
            self.client = create_client("assistant")
        except Exception as e:
            logger.error(f"Error initializing conversation generator client: {e}")
            raise

        # Initialize user simulator
        try:
            self.user_simulator = UserSimulator()
            logger.success(
                f"User simulator initialized (model: {self.user_simulator.model_name})"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize user simulator: {e}")
            self.user_simulator = None

    def _normalize_persona(self, persona):
        """Convert persona string to dict format if needed."""
        if persona is None:
            return random.choice(list(PERSONAS.values()))
        elif isinstance(persona, str):
            return PERSONAS.get(persona, PERSONAS["intermediate"])
        elif isinstance(persona, dict):
            return persona
        else:
            return PERSONAS["intermediate"]

    # ------------------------------------------------------------------
    # Shared prompt builders (Phase 2.2)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_initial_user_prompt(
        persona: dict, buggy_code: str, expected_errors: str
    ):
        """
        Build the system prompt and user instruction for the initial
        user message in both single-turn and multi-turn conversations.

        Returns:
            (system_prompt, user_instruction)
        """
        system_prompt = """You are in a 'Role Playing' scenario where you will be given a PERSONA of some LEVEL of a TypeScript developer who has encountered an error in your code and
you need help fixing it. You might also recieve the PROMPT STYLE of different expertise level. Your task will be presenting the bug as authentically as possible, showing the code and describing the error you're facing.

Additional Instructions for you:
- Be natural and authentic
- Show the code and describe the problem you're facing

The ideal conversational flow should be:
1. Roleplay as the user facing the bug
2. Present the buggy code and the error(s) you're encountering in a natural way as you are facing the error and asking for help
3. Your only task will be to generate the user's message asking for help, do not attempt to assist the user or provide any fixes. Just present the problem and ask for help in a natural way.

Generate ONLY and STRICTLY the user's message asking for help. Make sure to present the code and describe the error in a natural way.
"""

        user_instruction = f"""Your persona traits:
- {persona["persona"].capitalize()} level developer
- {persona["description"]}
- {persona["prompt_style"]}

You've encountered: {expected_errors} from this code:
```typescript
{buggy_code}
```

Generate ONLY the user's message (ROLE PLAY AS USER) as if you are facing the problem and asking for help.

IMPORTANT: Use the placeholder CODE_HERE where you want the buggy code to appear in your message. For example:
- "I'm facing an error: {expected_errors}. CODE_HERE Can you help?"
- "Can you check this code? CODE_HERE It's giving me errors."
- "CODE_HERE This isn't working and I'm getting: {expected_errors}"

Be authentic to your persona level. Use CODE_HERE as placeholder for code placement.
"""
        return system_prompt, user_instruction

    def _process_user_simulator_response(
        self, description: str, buggy_code: str
    ) -> str:
        """
        Process the user-simulator response: replace CODE_HERE placeholder with
        actual code or, if missing, strip TypeScript code blocks the model may
        have included and append the real code.

        Returns:
            Final user prompt string.
        """
        if "CODE_HERE" in description:
            user_prompt = description.replace(
                "CODE_HERE", f"\n\n```typescript\n{buggy_code}\n```\n"
            )
            logger.debug("User simulator response received (used CODE_HERE placeholder)")
        else:
            # Only strip typescript/ts-tagged code blocks; preserve other content
            description = re.sub(
                r"```(?:typescript|ts)[\s\S]*?```", "", description
            ).strip()
            user_prompt = (
                f"{description}\n\nHere's my code:\n\n```typescript\n{buggy_code}\n```"
            )
            logger.debug(
                "User simulator response received (no placeholder, appended code at end)"
            )
        return user_prompt

    # ------------------------------------------------------------------
    # Single-turn conversation
    # ------------------------------------------------------------------

    def generate_single_turn_synthetic(
        self, bug_data: Dict, persona: Optional[str] = None
    ) -> list[ChatCompletionMessageParam]:
        """
        Generate a single-turn conversation: user presents bug -> assistant fixes it.

        Args:
            bug_data: Dict with buggy_code and bug info (supports single or multiple bugs)
            persona: User persona (beginner/intermediate/advanced)

        Returns:
            List of messages in OpenAI format [user, assistant]
        """
        persona_info = self._normalize_persona(persona)

        buggy_code = bug_data.get("buggy_code", bug_data.get("code", ""))
        is_multiple = "bugs" in bug_data

        if is_multiple:
            expected_errors = bug_data.get(
                "expected_errors", "TypeScript compilation errors"
            )
        else:
            expected_errors = bug_data.get("expected_error", "TypeScript error")

        user_system_prompt, user_prompt_instruction = self._build_initial_user_prompt(
            persona_info, buggy_code, expected_errors
        )

        # Two-step generation with CODE_HERE placeholder
        try:
            if self.user_simulator:
                logger.debug(
                    f"Calling user simulator (model: {self.user_simulator.model_name})..."
                )
                description = self.user_simulator._call_model(
                    user_system_prompt, user_prompt_instruction
                )
                if description:
                    user_prompt = self._process_user_simulator_response(
                        description, buggy_code
                    )
                else:
                    logger.warning("User simulator returned empty, using fallback")
                    user_prompt = f"I'm getting this TypeScript error: {expected_errors}\n\nHere's my code:\n\n```typescript\n{buggy_code}\n```\n\nCan you help me fix it?"
            else:
                logger.warning("User simulator not available, using fallback")
                user_prompt = f"I'm getting this TypeScript error: {expected_errors}\n\nHere's my code:\n\n```typescript\n{buggy_code}\n```\n\nCan you help me fix it?"
        except Exception as e:
            logger.error(f"Error generating user prompt: {e}")
            user_prompt = f"I'm facing a TypeScript error: {expected_errors}\n\nHere's my code:\n\n```typescript\n{buggy_code}\n```\n\nHow do I fix this?"

        assistant_system_prompt = f"""You are an expert TypeScript developer and educator. Helping a {persona_info["persona"]} developer.
When a user presents buggy code:
1. Carefully analyze the code and identify ALL TypeScript errors
2. Explain why each error occurs (focus on TypeScript's type system)
3. Provide the complete corrected code
4. Explain what was changed and why
5. Be pedagogical and match your explanation depth to the user's level

Always provide working, corrected code."""

        try:
            api_messages: list[ChatCompletionMessageParam] = [
                {"role": "system", "content": assistant_system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            assistant_response = self.client.chat.completions.create(
                model=self.model,
                messages=api_messages,
            )

            assistant_message = (
                assistant_response.choices[0].message.content or ""
            ).strip()

            if not assistant_message:
                assistant_message = "I'll help you fix these TypeScript errors. [Error: No response generated]"

            return [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_message},
            ]

        except Exception as e:
            logger.error(f"Error generating assistant response: {e}")
            return [
                {"role": "user", "content": user_prompt},
                {
                    "role": "assistant",
                    "content": f"I can help you fix the TypeScript errors in this code. [Error: {str(e)}]",
                },
            ]

    # ------------------------------------------------------------------
    # Multi-turn conversation
    # ------------------------------------------------------------------

    def generate_multi_turn_synthetic(
        self,
        bug_data: Dict,
        persona: Optional[str] = None,
        num_turns: Optional[int] = None,
    ) -> list[ChatCompletionMessageParam]:
        """
        Generate a multi-turn verification-based conversation.
        User presents bug -> Assistant fixes -> User verifies -> Continue if issues found.

        Args:
            bug_data: Dict with buggy_code and bug info
            persona: User persona
            num_turns: Target number of complete turns (2-6)

        Returns:
            List of messages in OpenAI format [user, assistant, user, assistant, ...]
        """
        persona_info = self._normalize_persona(persona)

        if num_turns is None:
            num_turns = random.randint(2, 6)

        buggy_code = bug_data.get("buggy_code", bug_data.get("code", ""))
        is_multiple = "bugs" in bug_data

        if is_multiple:
            expected_errors = bug_data.get(
                "expected_errors", "TypeScript compilation errors"
            )
        else:
            expected_errors = bug_data.get("expected_error", "TypeScript error")

        messages: list[ChatCompletionMessageParam] = []

        try:
            # === TURN 1: User presents bug ===
            user_system_prompt, user_prompt_instruction = (
                self._build_initial_user_prompt(
                    persona_info, buggy_code, expected_errors
                )
            )

            if self.user_simulator:
                logger.debug(
                    f"Calling user simulator (model: {self.user_simulator.model_name})..."
                )
                description = self.user_simulator._call_model(
                    user_system_prompt, user_prompt_instruction
                )
                if description:
                    user_msg_1 = self._process_user_simulator_response(
                        description, buggy_code
                    )
                else:
                    logger.warning("User simulator returned empty, using fallback")
                    user_msg_1 = f"I'm getting this error: {expected_errors}\n\nHere's my code:\n\n```typescript\n{buggy_code}\n```\n\nCan you help?"
            else:
                logger.warning("User simulator not available, using fallback")
                user_msg_1 = f"I'm getting this error: {expected_errors}\n\nHere's my code:\n\n```typescript\n{buggy_code}\n```\n\nCan you help?"

            messages.append({"role": "user", "content": user_msg_1})

            # === TURN 1: Assistant provides fix ===
            assistant_system_prompt = f"""You are an expert TypeScript developer and educator. Helping a {persona_info["persona"]} developer.
When a user presents buggy code:
1. Carefully analyze the code and identify ALL TypeScript errors
2. Explain why each error occurs (focus on TypeScript's type system)
3. Provide the complete corrected code
4. Explain what was changed and why
5. Be pedagogical and match your explanation depth to the user's level

Always provide working, corrected code."""

            system_msg: list[ChatCompletionMessageParam] = [
                {"role": "system", "content": assistant_system_prompt}
            ]
            assistant_msg_1 = (
                self.client.chat.completions.create(
                    model=self.model,
                    messages=system_msg + messages,
                    )
                .choices[0]
                .message.content
                or ""
            ).strip()

            if not assistant_msg_1:
                assistant_msg_1 = "Let me help you fix these TypeScript errors."

            messages.append({"role": "assistant", "content": assistant_msg_1})

            # === FOLLOW-UP TURNS ===
            for turn_num in range(2, num_turns + 1):
                try:
                    if self.user_simulator:
                        logger.debug(
                            f"Calling user simulator for follow-up (turn {turn_num})..."
                        )
                        user_followup = self.user_simulator.generate_followup(
                            conversation_history=messages,
                            persona=persona_info["persona"],
                            intent=None,
                        )
                        if user_followup:
                            logger.debug("User simulator follow-up received")
                        else:
                            logger.debug("User satisfied with the answer")
                    else:
                        logger.warning(
                            "User simulator not available for follow-up, using fallback"
                        )
                        fallbacks = [
                            "Can you explain that part again?",
                            "I'm not sure I understand why that fixes it.",
                            "What about edge cases?",
                            "Is there an alternative approach?",
                            "Thanks! How would this handle null values?",
                        ]
                        user_followup = random.choice(fallbacks)

                    if not user_followup or len(user_followup) < 10:
                        if not user_followup:
                            logger.debug(
                                f"Conversation naturally concluded at turn {turn_num}"
                            )
                        break

                    messages.append({"role": "user", "content": user_followup})

                    assistant_followup = (
                        self.client.chat.completions.create(
                            model=self.model,
                            messages=system_msg + messages,
                                    )
                        .choices[0]
                        .message.content
                        or ""
                    ).strip()

                    if not assistant_followup:
                        assistant_followup = "Let me clarify that for you."

                    messages.append(
                        {"role": "assistant", "content": assistant_followup}
                    )

                except Exception as e:
                    logger.error(f"Error in turn {turn_num}: {e}")
                    if len(messages) % 2 != 0:
                        messages.append(
                            {
                                "role": "assistant",
                                "content": "I hope that helps clarify things!",
                            }
                        )
                    break

            # Ensure conversation ends with assistant
            if len(messages) % 2 != 0:
                messages.append(
                    {
                        "role": "assistant",
                        "content": "I hope that helps clarify things; Feel free to ask if you have more questions!",
                    }
                )

            return messages

        except Exception as e:
            logger.error(f"Error in multi-turn generation: {e}")
            if len(messages) == 0:
                return [
                    {
                        "role": "user",
                        "content": f"I have a TypeScript error: {expected_errors}",
                    },
                    {"role": "assistant", "content": "I'll help you fix that error."},
                ]
            elif len(messages) % 2 != 0:
                messages.append({"role": "assistant", "content": "I hope that helps!"})
            return messages

    # ------------------------------------------------------------------
    # Code generation conversation
    # ------------------------------------------------------------------

    def generate_code_generation_conversation(
        self,
        generated_code_data: Dict,
        persona: Optional[str] = None,
        num_turns: Optional[int] = None,
    ) -> list[ChatCompletionMessageParam]:
        """
        Generate a conversation about TypeScript code generation.
        User asks for code to be written, assistant generates it, user may ask for modifications.

        Args:
            generated_code_data: Dict from bug_injector.generate_typescript_code()
            persona: User persona
            num_turns: Target number of turns

        Returns:
            List of messages in OpenAI format
        """
        if persona is None:
            persona = random.choice(["beginner", "intermediate", "advanced"])

        if num_turns is None:
            num_turns = random.randint(2, 4)

        task = generated_code_data.get("task", "create TypeScript code")
        task_kind = generated_code_data.get("task_kind", "utility functions")
        generated_code = generated_code_data.get("code", "")

        messages: list[ChatCompletionMessageParam] = []

        try:
            generation_prompts = [
                f"Can you help me write TypeScript code for {task_kind}?",
                f"I need to implement something related to {task_kind} in TypeScript. Can you help?",
                f"Could you show me how to build {task_kind} using TypeScript?",
                f"I'm working on {task_kind}. Can you write some TypeScript code for this?",
            ]

            user_prompt_1 = random.choice(generation_prompts)
            messages.append({"role": "user", "content": user_prompt_1})

            system_prompt = f"""You are an expert TypeScript developer helping a {persona} developer.
Provide clean, well-typed TypeScript code with explanations."""

            assistant_msg_1 = f"""I'll help you with {task_kind}. Here's a TypeScript implementation for {task}:

```typescript
{generated_code}
```

This code {generated_code_data.get("description", "implements the requested functionality")}."""

            if generated_code_data.get("features"):
                assistant_msg_1 += "\n\nKey TypeScript features used:\n"
                for feature in generated_code_data["features"][:3]:
                    assistant_msg_1 += f"- {feature}\n"

            messages.append({"role": "assistant", "content": assistant_msg_1})

            if num_turns > 2:
                followup_intents = [
                    "ask_clarification",
                    "ask_alternative",
                    "request_more_info",
                ]

                for turn_num in range(num_turns - 2):
                    try:
                        if self.user_simulator:
                            user_followup = self.user_simulator.generate_followup(
                                conversation_history=messages,
                                persona=persona,
                                intent=random.choice(followup_intents),
                            )
                            if not user_followup:
                                logger.debug("Code gen conversation naturally concluded")
                                break
                        else:
                            user_followup = "Can you explain how this works?"

                        messages.append({"role": "user", "content": user_followup})

                        sys_msg: list[ChatCompletionMessageParam] = [
                            {"role": "system", "content": system_prompt}
                        ]
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=sys_msg + messages,
                                    )
                        assistant_response = response.choices[0].message.content or ""
                        messages.append(
                            {"role": "assistant", "content": assistant_response}
                        )

                    except Exception as e:
                        logger.error(f"Error generating code gen turn {turn_num + 3}: {e}")
                        break

            return messages

        except Exception as e:
            logger.error(f"Error in code generation conversation: {e}")
            task_kind = generated_code_data.get("task_kind", "TypeScript code")
            return [
                {"role": "user", "content": f"Can you help me with {task_kind}?"},
                {
                    "role": "assistant",
                    "content": f"I'd be happy to help with that TypeScript code. [Error: {str(e)}]",
                },
            ]
