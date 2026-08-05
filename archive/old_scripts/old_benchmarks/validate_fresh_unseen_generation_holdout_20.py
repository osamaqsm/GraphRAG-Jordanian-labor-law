from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


DIACRITICS_RE = re.compile(
    r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_tokens(text: str) -> set[str]:
    value = str(text).lower()
    value = DIACRITICS_RE.sub("", value)
    value = re.sub(r"[إأآٱ]", "ا", value)
    value = (
        value.replace("ى", "ي")
        .replace("ة", "ه")
        .replace("ؤ", "و")
        .replace("ئ", "ي")
    )
    value = re.sub(r"[^\u0600-\u06ff0-9]+", " ", value)
    return set(value.split())


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that the fresh generation holdout does not overlap "
            "the previous master retrieval benchmark."
        )
    )
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument(
        "--max-question-jaccard",
        type=float,
        default=0.45,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    holdout = json.loads(
        args.holdout.read_text(encoding="utf-8-sig")
    )
    master = json.loads(
        args.master.read_text(encoding="utf-8-sig")
    )

    cases = list(holdout.get("cases", []))
    assert len(cases) == 20
    assert len({case["id"] for case in cases}) == 20

    distribution: dict[str, int] = {}
    for case in cases:
        behavior = str(case["expected_behavior"])
        distribution[behavior] = distribution.get(behavior, 0) + 1

    assert distribution == {
        "retrieve": 16,
        "clarify": 2,
        "abstain": 2,
    }

    previous_articles = {
        int(value)
        for value in master.get("unique_gold_articles", [])
    }
    new_articles = {
        int(number)
        for case in cases
        if case["expected_behavior"] == "retrieve"
        for number in case.get("articles", [])
    }
    overlap = sorted(previous_articles & new_articles)
    if overlap:
        raise RuntimeError(
            f"Gold article overlap with master benchmark: {overlap}"
        )

    old_questions = list(master.get("questions", []))
    old_exact = {
        " ".join(sorted(normalize_tokens(item["question"])))
        for item in old_questions
    }

    near_duplicates: list[dict[str, Any]] = []
    for case in cases:
        question = str(case["question"])
        tokens = normalize_tokens(question)
        exact_key = " ".join(sorted(tokens))
        if exact_key in old_exact:
            raise RuntimeError(
                f"Exact normalized question overlap: {case['id']}"
            )

        best_score = 0.0
        best_id = ""
        best_question = ""
        for old in old_questions:
            score = jaccard(
                tokens,
                normalize_tokens(old["question"]),
            )
            if score > best_score:
                best_score = score
                best_id = str(old["id"])
                best_question = str(old["question"])

        if best_score >= args.max_question_jaccard:
            near_duplicates.append(
                {
                    "new_id": case["id"],
                    "old_id": best_id,
                    "score": round(best_score, 4),
                    "new_question": question,
                    "old_question": best_question,
                }
            )

    if near_duplicates:
        raise RuntimeError(
            "Near-duplicate questions found:\n"
            + json.dumps(
                near_duplicates,
                ensure_ascii=False,
                indent=2,
            )
        )

    print("Fresh unseen generation holdout validation passed.")
    print("Questions: 20")
    print(f"Behavior distribution: {distribution}")
    print(f"New retrieve gold articles: {len(new_articles)}")
    print("Gold article overlap with previous 120: 0")
    print("Exact question overlap with previous 120: 0")
    print(
        "Maximum allowed normalized Jaccard similarity: "
        f"{args.max_question_jaccard}"
    )
    print(f"Holdout SHA-256: {sha256_file(args.holdout)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
