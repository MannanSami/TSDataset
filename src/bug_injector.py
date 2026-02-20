import json
import os
from typing import Dict, Optional

from config import create_client, extract_json_from_response, logger


class BugInjector:
    """Injects TypeScript-specific bugs into clean code using LLM."""

    TS_BUG_CATEGORIES = [
        "type-mismatch",
        "generic-constraint",
        "union-type-error",
        "null-undefined",
        "async-promise",
        "interface-mismatch",
        "readonly-modifier",
        "tuple-type",
        "type-inference-error",
        "optional-property",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize the bug injector with OpenAI client."""
        resolved_model = model or os.getenv("ASSISTANT_MODEL")
        if not resolved_model:
            raise ValueError(
                "No model specified. Set ASSISTANT_MODEL env var or pass model parameter."
            )
        self.model: str = resolved_model

        try:
            self.client = create_client("assistant")
            self.base_url = self.client.base_url
        except Exception as e:
            logger.error(f"Error initializing bug injector client: {e}")
            raise

    def inject_bug(
        self, clean_code: str, bug_category: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Inject a TypeScript-specific bug into clean code.

        Args:
            clean_code: The original TypeScript code
            bug_category: Optional specific bug category to inject

        Returns:
            Dict with keys: buggy_code, bug_description, expected_error, bug_category
        """
        import random

        if bug_category is None:
            bug_category = random.choice(self.TS_BUG_CATEGORIES)

        prompt = f"""You are a TypeScript specialist who excels at introducing realistic TypeScript-specific bugs.

TASK: Introduce a realistic {bug_category} bug into the following TypeScript code.

REQUIREMENTS:
1. The bug MUST be TypeScript-specific (type system related)
2. The bug should be realistic - something a developer might actually write
3. The bug MUST cause a compile-time error (not runtime)
4. Only introduce ONE clear bug
5. The bug should be fixable through reasoning
6. Keep the overall code structure intact

ORIGINAL CODE:
```typescript
{clean_code}
```

Return your response in this EXACT JSON format (no markdown, just raw JSON):
{{
  "buggyCode": "the full code with the bug introduced",
  "bugDescription": "concise description of what bug was introduced",
  "expectedError": "the expected TypeScript compiler error message"
}}"""
        content = ""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that introduces specific TypeScript bugs. Always respond with valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            content = (response.choices[0].message.content or "").strip()
            content = extract_json_from_response(content)

            result = json.loads(content)

            return {
                "buggy_code": result.get("buggyCode", clean_code),
                "bug_description": result.get("bugDescription", "Unknown bug"),
                "expected_error": result.get("expectedError", "Unknown error"),
                "bug_category": bug_category,
            }

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.debug(f"Content received: {content[:500]}")
            return {
                "buggy_code": clean_code,
                "bug_description": "Failed to inject bug: JSON parse error",
                "expected_error": "N/A",
                "bug_category": bug_category,
            }
        except Exception as e:
            logger.error(f"Error in bug injection: {e}")
            return {
                "buggy_code": clean_code,
                "bug_description": f"Failed to inject bug: {str(e)}",
                "expected_error": "N/A",
                "bug_category": bug_category,
            }

    def inject_multiple_bugs(
        self, clean_code: str, num_bugs: int = 2, bug_categories: Optional[list] = None
    ) -> Dict:
        """
        Inject multiple TypeScript-specific bugs into clean code.

        Args:
            clean_code: The original TypeScript code
            num_bugs: Number of bugs to inject (2-5 recommended)
            bug_categories: Optional list of specific bug categories to inject

        Returns:
            Dict with keys: buggy_code, bugs (list of bug info), all_bug_categories
        """
        import random

        if bug_categories is None:
            bug_categories = random.sample(
                self.TS_BUG_CATEGORIES, min(num_bugs, len(self.TS_BUG_CATEGORIES))
            )

        categories_str = ", ".join(bug_categories[:num_bugs])

        prompt = f"""You are a TypeScript specialist who excels at introducing realistic TypeScript-specific bugs.

TASK: Introduce {num_bugs} DIFFERENT realistic TypeScript bugs into the following code.

BUG CATEGORIES TO INJECT: {categories_str}

REQUIREMENTS:
1. Introduce exactly {num_bugs} distinct bugs (one from each category listed)
2. Each bug MUST be TypeScript-specific (type system related)
3. All bugs should be realistic - something a developer might actually write
4. All bugs MUST cause compile-time errors (not runtime)
5. Make bugs independent where possible (but they can interact)
6. Keep the overall code structure intact
7. Bugs should be fixable through reasoning

ORIGINAL CODE:
```typescript
{clean_code}
```

Return your response in this EXACT JSON format (no markdown, just raw JSON):
{{
  "buggyCode": "the full code with all {num_bugs} bugs introduced",
  "bugs": [
    {{
      "category": "bug category 1",
      "description": "description of bug 1",
      "location": "where in the code (e.g., line reference or function name)"
    }},
    {{
      "category": "bug category 2",
      "description": "description of bug 2",
      "location": "where in the code"
    }}
  ],
  "expectedErrors": "the expected TypeScript compiler error messages (can be multiple)"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that introduces specific TypeScript bugs. Always respond with valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            content = (response.choices[0].message.content or "").strip()
            content = extract_json_from_response(content)

            result = json.loads(content)

            return {
                "buggy_code": result.get("buggyCode", clean_code),
                "bugs": result.get("bugs", []),
                "expected_errors": result.get(
                    "expectedErrors", "Multiple TypeScript errors"
                ),
                "all_bug_categories": bug_categories[:num_bugs],
                "num_bugs": num_bugs,
            }

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in multiple bug injection: {e}")
            # Fallback to single bug
            return self._fallback_multiple_bugs(clean_code, num_bugs, bug_categories)
        except Exception as e:
            logger.error(f"Error in multiple bug injection: {e}")
            return self._fallback_multiple_bugs(clean_code, num_bugs, bug_categories)

    def generate_typescript_code(self, task_kind: str = "utility functions") -> Dict:
        """
        Generate TypeScript code for a dynamically created task.

        Args:
            task_kind: Category/kind of task (e.g., "data structures", "utility functions")

        Returns:
            Dict with keys: code, description, features, task
        """

        prompt = f"""You are an expert TypeScript developer. Generate a complete, production-quality TypeScript code example.

TASK CATEGORY: {task_kind}

REQUIREMENTS:
1. First, come up with a SPECIFIC task within the "{task_kind}" category
2. Write clean, well-typed TypeScript code (50-200 lines)
3. Use TypeScript-specific features (interfaces, types, generics where appropriate)
4. Code should be complete and compilable
5. Include proper type annotations throughout
6. Add brief comments for clarity
7. Follow TypeScript best practices
8. Code should demonstrate real-world usage patterns

Return your response in this EXACT JSON format (no markdown, just raw JSON):
{{
  "task": "specific description of what you decided to implement",
  "code": "the complete TypeScript code (50-200 lines)",
  "description": "brief description of what the code does",
  "features": ["TypeScript feature 1 used", "TypeScript feature 2 used", "..."]
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert TypeScript developer. Always respond with valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            content = (response.choices[0].message.content or "").strip()
            content = extract_json_from_response(content)

            result = json.loads(content)

            return {
                "code": result.get("code", "// Failed to generate code"),
                "description": result.get(
                    "description", f"TypeScript code for {task_kind}"
                ),
                "features": result.get("features", []),
                "task": result.get("task", f"Generate {task_kind}"),
                "task_kind": task_kind,
            }

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in code generation: {e}")
            return {
                "code": "// Failed to generate code: JSON parse error",
                "description": f"TypeScript code for {task_kind}",
                "features": [],
                "task": f"Generate {task_kind}",
                "task_kind": task_kind,
            }
        except Exception as e:
            logger.error(f"Error in code generation: {e}")
            return {
                "code": f"// Failed to generate code: {str(e)}",
                "description": f"TypeScript code for {task_kind}",
                "features": [],
                "task": f"Generate {task_kind}",
                "task_kind": task_kind,
            }

    def _fallback_multiple_bugs(
        self, clean_code: str, num_bugs: int, bug_categories: list
    ) -> Dict:
        """Fallback when multiple bug injection fails - try single bugs sequentially."""
        logger.warning("Using fallback: injecting bugs one by one")
        current_code = clean_code
        all_bugs = []

        for i, category in enumerate(bug_categories[:num_bugs]):
            try:
                bug_result = self.inject_bug(current_code, category)
                if bug_result["buggy_code"] != current_code:
                    current_code = bug_result["buggy_code"]
                    all_bugs.append(
                        {
                            "category": category,
                            "description": bug_result["bug_description"],
                            "location": f"bug {i + 1}",
                        }
                    )
            except Exception as e:
                logger.error(f"Failed to inject bug {i + 1}: {e}")
                continue

        return {
            "buggy_code": current_code,
            "bugs": all_bugs,
            "expected_errors": "Multiple TypeScript errors",
            "all_bug_categories": bug_categories[:num_bugs],
            "num_bugs": len(all_bugs),
        }
