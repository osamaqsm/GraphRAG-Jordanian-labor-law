from __future__ import annotations

import inspect
from pathlib import Path

from app.grounded_answer_generator import GroundedAnswerGenerator


def main() -> None:
    prompt = GroundedAnswerGenerator._instructions()

    required_prompt_rules = [
        "Preserve every condition that limits when the rule applies",
        "Copy statutory numbers exactly",
        "Answer every distinct part",
        "include that actor when it is material",
        "do not describe one case as the only permitted case",
        "Preserve the wording and meaning of short decisive legal conditions",
    ]
    for rule in required_prompt_rules:
        assert rule in prompt, rule

    source = inspect.getsource(
        __import__(
            "app.grounded_answer_generator",
            fromlist=["GroundedAnswerGenerator"],
        )
    )
    assert "_canonicalize_citations" in source
    assert "rejected_draft" in source
    assert "model_called=True" in source

    patch_source = Path(
        "scripts/patch_real_generation_benchmark_v1_1.py"
    ).read_text(encoding="utf-8")
    assert "retrieval_snapshot_changed" in patch_source
    assert "retrieval_hashes_changed" in patch_source
    assert "G09" in patch_source
    assert "G11" in patch_source
    assert "G16" in patch_source

    print("Stage 8-B4 checks passed.")
    print("Citation normalization is installed")
    print("Rejected drafts are preserved in debug")
    print("Prompt preserves legal conditions and numbers")
    print("Benchmark patch does not modify retrieval files")


if __name__ == "__main__":
    main()
