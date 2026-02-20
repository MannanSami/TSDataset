from config import logger
from validators import (
    compute_quality_score,
    validate_bug_fixing_sample,
    validate_code_gen_sample,
)


def main():
    logger.info("Testing validators...")

    # ── Bug-fixing: good sample ──
    good_bug = {
        "conversation": [
            {
                "role": "user",
                "content": "Help me fix this:\n```typescript\nlet x: string = 5;\n```",
            },
            {
                "role": "assistant",
                "content": "Here is the fix:\n```typescript\nlet x: number = 5;\n```",
            },
        ],
        "metadata": {
            "buggy_code": "let x: string = 5;",
            "original_code": "let x: number = 5;",
        },
    }
    valid, issues = validate_bug_fixing_sample(good_bug)
    assert valid and issues == [], f"FAIL good_bug: {issues}"
    logger.success("validate_bug_fixing_sample: good sample passes")

    # ── Bug-fixing: bad sample (error marker, identical code) ──
    bad_bug = {
        "conversation": [
            {
                "role": "user",
                "content": "Help me fix this:\n```typescript\nlet x: string = 5;\n```",
            },
            {"role": "assistant", "content": "[Error: timeout]"},
        ],
        "metadata": {
            "buggy_code": "let x: string = 5;",
            "original_code": "let x: string = 5;",
        },
    }
    valid, issues = validate_bug_fixing_sample(bad_bug)
    assert not valid, "FAIL bad_bug should be invalid"
    assert len(issues) >= 2, f"FAIL bad_bug expected >=2 issues, got {issues}"
    logger.success(f"validate_bug_fixing_sample: bad sample caught {len(issues)} issues")

    # ── Code-gen: good sample ──
    good_gen = {
        "conversation": [
            {"role": "user", "content": "Write an add function."},
            {
                "role": "assistant",
                "content": "```typescript\nfunction add(a: number, b: number) { return a + b; }\n```",
            },
        ],
        "metadata": {
            "generated_code": "function add(a: number, b: number) { return a + b; }"
        },
    }
    valid, issues = validate_code_gen_sample(good_gen)
    assert valid and issues == [], f"FAIL good_gen: {issues}"
    logger.success("validate_code_gen_sample: good sample passes")

    # ── Code-gen: bad sample (Failed code) ──
    bad_gen = {
        "conversation": [
            {"role": "user", "content": "Write something."},
            {"role": "assistant", "content": "Sorry, no code."},
        ],
        "metadata": {"generated_code": "// Failed to generate code"},
    }
    valid, issues = validate_code_gen_sample(bad_gen)
    assert not valid, "FAIL bad_gen should be invalid"
    logger.success(f"validate_code_gen_sample: bad sample caught {len(issues)} issues")

    # ── Quality scores ──
    good_score = compute_quality_score(good_bug)
    bad_score = compute_quality_score(bad_bug)
    assert 0.0 <= good_score <= 1.0
    assert 0.0 <= bad_score <= 1.0
    assert good_score > bad_score, (
        f"FAIL good ({good_score}) should beat bad ({bad_score})"
    )
    logger.success(f"compute_quality_score: good={good_score:.2f}, bad={bad_score:.2f}")

    logger.success("All validator tests passed!")


if __name__ == "__main__":
    main()
