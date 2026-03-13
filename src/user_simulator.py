import random
import re
from typing import Optional

from config import PERSONAS, logger
from opencode_client import call_opencode


class UserSimulator:
    """Simulates user interactions with different personas and skill levels."""

    def __init__(self, attach_url: str, model: Optional[str] = None):
        """Initialize the user simulator with opencode attach URL."""
        self.attach_url = attach_url
        self.model_name = model or "opencode/minimax-m2.5-free"
        logger.info(f"UserSimulator initialized (attach_url: {self.attach_url})")

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """Remove <think></think> tags from model output."""
        if not text:
            return text
        cleaned = re.sub(
            r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE
        )
        cleaned = re.sub(r"\n\s*\n\s*\n", "\n\n", cleaned)
        return cleaned.strip()

    async def _call_model(
        self,
        system_prompt: str,
        user_prompt: str,
        timeout: int = 180,
        max_retries: int = 3,
    ) -> Optional[str]:
        """Call opencode to generate a response with retries."""
        try:
            output = await call_opencode(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                attach_url=self.attach_url,
                timeout=timeout,
                max_retries=max_retries,
            )
            output = self._strip_think_tags(output)
            return output
        except Exception as e:
            logger.error(f"Model call failed: {e}")
            return None

    def _get_persona_context(self, persona: str) -> str:
        """Get the context description for a persona."""
        persona_info = PERSONAS.get(persona, PERSONAS["intermediate"])
        context = f"{persona_info['description']}. Traits:\n"
        for trait in persona_info["traits"]:
            context += f"- {trait}\n"
        return context

    async def generate_initial_prompt(
        self,
        code: str,
        scenario: str = "bug_fixing",
        persona: str = "intermediate",
        context: Optional[str] = None,
    ) -> str:
        """
        Generate an initial user prompt based on scenario and persona.

        Args:
            code: The TypeScript code (buggy or for discussion)
            scenario: 'bug_fixing', 'code_generation', 'code_review', 'explanation'
            persona: 'beginner', 'intermediate', or 'advanced'
            context: Optional additional context about the bug or task

        Returns:
            Generated user prompt string
        """
        persona_context = self._get_persona_context(persona)

        scenario_instructions = {
            "bug_fixing": f"Generate a natural user message from a {persona} TypeScript developer who has encountered an error in their code. They should present the buggy code and ask for help. The message should feel authentic and match the persona's skill level.",
            "code_generation": f"Generate a natural user message from a {persona} TypeScript developer who wants help writing TypeScript code for a specific task. They should describe what they want to build.",
            "code_review": f"Generate a natural user message from a {persona} TypeScript developer who wants feedback on their code. They should present the code and ask for review or improvements.",
            "explanation": f"Generate a natural user message from a {persona} TypeScript developer who wants to understand how some TypeScript code works.",
        }

        instruction = scenario_instructions.get(
            scenario, scenario_instructions["bug_fixing"]
        )

        system_prompt = f"""You are simulating a {persona} TypeScript developer in a conversation.
{persona_context}

Important guidelines:
- Generate ONLY the user's message, nothing else
- Make it natural and conversational
- Match the persona's skill level and traits
- Keep it concise (2-4 sentences typically)
- Use CODE_HERE as a placeholder where you want your code to appear
- The user should sound authentic, not like an AI"""

        user_prompt = f"""{instruction}

{f"Context about the code/bug: {context}" if context else ""}

TypeScript code:
```typescript
{code}
```

Generate the user's message:"""

        response = await self._call_model(system_prompt, user_prompt)
        if response:
            return response

        return self._fallback_prompt(code, scenario, persona)

    async def generate_followup(
        self,
        conversation_history: list[dict],
        persona: str = "intermediate",
        intent: Optional[str] = None,
    ) -> str | None:
        """
        Generate a follow-up user message based on conversation history.

        Args:
            conversation_history: Previous messages in the conversation
            persona: User persona
            intent: Optional intent like 'ask_clarification', 'request_fix', 'ask_alternative', etc.

        Returns:
            Generated follow-up message or None if conversation should end
        """
        persona_context = self._get_persona_context(persona)

        if not intent:
            intent = random.choice(
                [
                    "ask_clarification",
                    "request_fix",
                    "ask_why",
                    "ask_alternative",
                    "express_thanks_continue",
                    "ask_edge_case",
                    "ask_about_what_you_are_confused_about",
                ]
            )

        intent_guidance = {
            "ask_clarification": "Ask for clarification about something in the assistant's response",
            "request_fix": "Ask to see the corrected/fixed code",
            "ask_why": "Ask why something works a certain way or why the error occurred",
            "ask_alternative": "Ask if there's an alternative approach",
            "express_thanks_continue": "Thank the assistant and ask a related question",
            "ask_edge_case": "Ask about an edge case or related scenario",
            "request_more_info": "Request more details or examples",
            "ask_about_what_you_are_confused_about": "Ask about the specific part of the explanation or code that you are confused about or is confusing.",
        }

        prompt_style = random.choice(["role_play", "intent_based", "natural"])

        if prompt_style == "role_play":
            system_prompt = f"""You are in a 'Role Playing' scenario. You are a {persona} TypeScript developer.

Your persona traits:
{persona_context}

The assistant just provided a response. Now you need to:
1. Review the assistant's response as a {persona} developer would
2. If you spot issues or don't understand something, ask about it naturally
3. If you want clarification, ask naturally
4. If you're satisfied and have no more questions, output "CONVERSATION_END"
5. Be natural and authentic to your persona level

Current intent: {intent_guidance.get(intent, "Continue naturally")}

Generate ONLY your next message as the user. Do not provide solutions or act as the assistant."""

        elif prompt_style == "intent_based":
            system_prompt = f"""You are simulating a {persona} TypeScript developer in an ongoing conversation.
{persona_context}

Current goal: {intent_guidance.get(intent, "Continue the conversation naturally")}

Guidelines:
- Generate ONLY the user's next message based on the conversation
- Keep it natural and conversational (1-3 sentences typically)
- Match the persona's skill level
- DO NOT repeat previous questions - be creative and varied
- If asking for clarification, ask about DIFFERENT aspects each time
- Be specific about what confuses you (avoid generic "I don't understand")
- If the assistant has fully answered and you have no concerns, output "CONVERSATION_END"
- Only continue if you genuinely need clarification"""

        else:  # natural style
            system_prompt = f"""You are a {persona} TypeScript developer having a conversation.
{persona_context}

Based on what the assistant just said:
- If something is unclear at your level, ask about it
- If you see potential issues, mention them
- If you want to understand deeper, ask why
- You can ask about alternatives or best practices
- If you're satisfied and everything is clear, output "CONVERSATION_END"

Intent suggestion: {intent_guidance.get(intent, "Continue naturally")}

Keep it natural (1-3 sentences). Match your persona. Be specific, not generic."""

        conversation_text = "\n\n".join(
            [
                f"{msg['role'].upper()}: {str(msg.get('content', ''))}"
                for msg in conversation_history
            ]
        )

        user_prompt = f"""Based on this conversation so far:

{conversation_text}

Generate your next message as the user.

If the assistant has fully resolved your issue and explained everything clearly, and you have no more questions, output "CONVERSATION_END".
Otherwise, generate your next natural message:"""

        response = await self._call_model(system_prompt, user_prompt)
        if response:
            # Hardened CONVERSATION_END detection (Phase 3.4)
            if "CONVERSATION_END" in response.strip().upper():
                return None
            # Detect short "thanks" / satisfaction responses with no question
            if (
                re.match(
                    r"^(thanks|thank you|got it|perfect|great|awesome|that (helps|makes sense)|understood)[\.\!\s]*$",
                    response.strip(),
                    re.IGNORECASE,
                )
                and "?" not in response
            ):
                return None
            return response

        return self._fallback_followup(intent, persona)

    def _fallback_prompt(self, code: str, scenario: str, persona: str) -> str:
        """Fallback prompts if model is unavailable."""
        templates = {
            "bug_fixing": [
                f"I'm getting a TypeScript error in this code:\n\n```typescript\n{code}\n```\n\nWhat's wrong with it?",
                f"This TypeScript code isn't compiling. Can you help me fix it?\n\n```typescript\n{code}\n```",
                f"I have a type error in this code but I'm not sure why:\n\n```typescript\n{code}\n```",
            ],
            "code_generation": [
                "Can you help me write TypeScript code for this task?",
                "I need to implement this functionality in TypeScript. How should I approach it?",
                "Could you show me how to write this in TypeScript?",
            ],
        }

        template_list = templates.get(scenario, templates["bug_fixing"])
        return random.choice(template_list)

    def _fallback_followup(self, intent: str, persona: str) -> str:
        """Fallback follow-up messages if model is unavailable."""
        fallbacks = {
            "ask_clarification": "Can you explain that part again?",
            "request_fix": "Can you show me the corrected code?",
            "ask_why": "Why does that cause an error?",
            "ask_alternative": "Is there another way to do this?",
            "express_thanks_continue": "Thanks! How would this work with other types?",
            "ask_edge_case": "What about edge cases?",
        }

        return fallbacks.get(intent, "Can you explain more?")
