from __future__ import annotations

import inspect

import app.grounded_answer_generator as module
from app.grounded_answer_generator import GroundedAnswerGenerator


def main() -> None:
    source = inspect.getsource(module).lower()

    forbidden = [
        "retrievalonlypipeline",
        "retrievalservice",
        "import weaviate",
        "graphtraversal",
        "embeddings.create",
        "selection_plan",
        "verification_model",
        "_selection_instructions",
        "_verification_instructions",
    ]
    for value in forbidden:
        assert value not in source, value

    assert source.count("responses.create") == 1
    prompt = GroundedAnswerGenerator._instructions()
    assert "user_question" in prompt
    assert "retrieval_result" in prompt
    assert "retrieval_result.articles[].text" in prompt

    print("Simple Step 8 architecture checks passed.")
    print("Retrieval dependencies: 0")
    print("Evidence-selection calls: 0")
    print("Verification calls: 0")
    print("Responses API calls on retrieve route: 1")


if __name__ == "__main__":
    main()
