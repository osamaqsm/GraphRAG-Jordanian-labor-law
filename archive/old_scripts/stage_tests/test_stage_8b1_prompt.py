from __future__ import annotations

from app.grounded_answer_generator import GroundedAnswerGenerator


def main() -> None:
    prompt = " ".join(
        GroundedAnswerGenerator._instructions().split()
    )

    required = [
        "Start immediately with the legal rule or direct answer",
        'Do not add a separate source line',
        'heading named "المصدر"',
        "For a simple question answered by one rule, return an empty list",
        "Return an empty list when the supplied provisions answer the question",
        "Never include generic boilerplate",
        "Place the citation immediately after the sentence or clause it supports",
    ]
    for clause in required:
        assert clause in prompt, clause

    forbidden_prompt_defaults = [
        "always add a legal disclaimer",
        "always include limitations",
        "always provide key points",
    ]
    lowered = prompt.lower()
    for clause in forbidden_prompt_defaults:
        assert clause not in lowered, clause

    print("Stage 8-B1 prompt checks passed.")
    print("Direct concise Arabic answer required")
    print("No separate source line")
    print("Simple answers use empty key_points")
    print("Sufficient evidence uses empty limitations")


if __name__ == "__main__":
    main()
