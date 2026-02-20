from typing import Dict, List, Tuple


def validate_bug_fixing_sample(entry: Dict) -> Tuple[bool, List[str]]:
    """
    Validate a bug-fixing dataset sample.

    Returns:
        (is_valid, issues): A bool and a list of issue descriptions.
    """
    issues = []
    metadata = entry.get("metadata", {})
    conversation = entry.get("conversation", [])

    # buggy_code must differ from original_code
    if metadata.get("buggy_code") and metadata.get("original_code"):
        if metadata["buggy_code"] == metadata["original_code"]:
            issues.append("buggy_code is identical to original_code")

    # Conversation must start with user and end with assistant
    if conversation:
        if conversation[0].get("role") != "user":
            issues.append("conversation does not start with a user message")
        if conversation[-1].get("role") != "assistant":
            issues.append("conversation does not end with an assistant message")
    else:
        issues.append("conversation is empty")

    # At least one assistant response should contain a code block
    has_code_block = any(
        "```" in msg.get("content", "")
        for msg in conversation
        if msg.get("role") == "assistant"
    )
    if not has_code_block:
        issues.append("no assistant response contains a code block")

    # User message should contain the buggy code
    buggy_code = metadata.get("buggy_code", "")
    if buggy_code and conversation:
        user_messages = " ".join(
            msg.get("content", "") for msg in conversation if msg.get("role") == "user"
        )
        if buggy_code not in user_messages:
            issues.append("user message does not contain the buggy code")

    # No [Error: markers in assistant responses
    for msg in conversation:
        if msg.get("role") == "assistant" and "[Error:" in msg.get("content", ""):
            issues.append("assistant response contains an [Error: marker")
            break

    # Detect hallucinated truncation claims
    truncation_phrases = ["cut off", "truncated", "incomplete", "continue where you left off", "finish your response"]
    for msg in conversation:
        if msg.get("role") == "user":
            content_lower = msg.get("content", "").lower()
            if any(phrase in content_lower for phrase in truncation_phrases):
                issues.append("user message contains hallucinated truncation claim")
                break

    return (len(issues) == 0, issues)


def validate_code_gen_sample(entry: Dict) -> Tuple[bool, List[str]]:
    """
    Validate a code-generation dataset sample.

    Returns:
        (is_valid, issues): A bool and a list of issue descriptions.
    """
    issues = []
    metadata = entry.get("metadata", {})
    conversation = entry.get("conversation", [])

    generated_code = metadata.get("generated_code", "")
    if not generated_code or "Failed" in generated_code:
        issues.append("generated_code is missing or contains 'Failed'")

    # At least one assistant response should contain a code block
    has_code_block = any(
        "```" in msg.get("content", "")
        for msg in conversation
        if msg.get("role") == "assistant"
    )
    if not has_code_block:
        issues.append("no assistant response contains a code block")

    # No [Error: markers in assistant responses
    for msg in conversation:
        if msg.get("role") == "assistant" and "[Error:" in msg.get("content", ""):
            issues.append("assistant response contains an [Error: marker")
            break

    # Detect hallucinated truncation claims
    truncation_phrases = ["cut off", "truncated", "incomplete", "continue where you left off", "finish your response"]
    for msg in conversation:
        if msg.get("role") == "user":
            content_lower = msg.get("content", "").lower()
            if any(phrase in content_lower for phrase in truncation_phrases):
                issues.append("user message contains hallucinated truncation claim")
                break

    return (len(issues) == 0, issues)


def compute_quality_score(entry: Dict) -> float:
    """
    Compute a quality score for a dataset sample.

    Returns:
        A float between 0.0 and 1.0.
    """
    score = 0.5  # baseline
    metadata = entry.get("metadata", {})
    conversation = entry.get("conversation", [])

    # Penalize error markers in assistant responses
    for msg in conversation:
        if msg.get("role") == "assistant" and "[Error:" in msg.get("content", ""):
            score -= 0.3
            break

    # Penalize very short assistant responses
    for msg in conversation:
        if msg.get("role") == "assistant" and len(msg.get("content", "")) < 50:
            score -= 0.1
            break

    # Reward longer conversations (more than 2 messages)
    if len(conversation) > 2:
        score += 0.1

    # Reward buggy_code != original_code (for bug-fixing samples)
    if metadata.get("buggy_code") and metadata.get("original_code"):
        if metadata["buggy_code"] != metadata["original_code"]:
            score += 0.1

    # Reward code blocks in assistant responses
    has_code = any(
        "```" in msg.get("content", "")
        for msg in conversation
        if msg.get("role") == "assistant"
    )
    if has_code:
        score += 0.2

    # Truncation hallucination penalty
    truncation_phrases = ["cut off", "truncated", "incomplete", "continue where you left off", "finish your response"]
    has_truncation_claim = False
    for msg in conversation:
        if msg.get("role") == "user":
            content_lower = msg.get("content", "").lower()
            if any(phrase in content_lower for phrase in truncation_phrases):
                has_truncation_claim = True
                score -= 0.3
                break

    # Metadata mismatch penalty
    is_multi_turn = metadata.get("is_multi_turn", False)
    num_turns = metadata.get("num_turns", 0)
    if (is_multi_turn and num_turns == 1) or (not is_multi_turn and num_turns > 1):
        score -= 0.2

    # Conversation coherence reward
    user_followups = [
        msg for msg in conversation[1:] if msg.get("role") == "user"
    ]
    if user_followups and all(
        len(msg.get("content", "")) > 20 for msg in user_followups
    ) and not has_truncation_claim:
        score += 0.1

    return max(0.0, min(1.0, score))
