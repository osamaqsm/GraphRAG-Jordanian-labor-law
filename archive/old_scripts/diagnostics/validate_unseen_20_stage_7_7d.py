from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from rdflib import Graph, Namespace, RDFS

DEFAULT_BENCHMARK = Path(
    "/app/data/benchmarks/retrieval_benchmark_unseen_20_stage_7_7d.json"
)
DEFAULT_TTL = Path("/app/data/jordan_labor_law_full_knowledge_graph.ttl")
DEFAULT_DEV20 = Path("/app/data/benchmarks/retrieval_benchmark_20.json")
DEFAULT_PREVIOUS50 = Path(
    "/app/data/benchmarks/retrieval_benchmark_unseen_50.json"
)
DEFAULT_PREVIOUS30 = Path(
    "/app/data/benchmarks/retrieval_benchmark_unseen_30_final.json"
)

EXPECTED_DISTRIBUTION = {
    "straightforward": 4,
    "paraphrased": 3,
    "typo": 2,
    "colloquial": 2,
    "numerical": 2,
    "multi_article": 1,
    "ambiguous": 3,
    "out_of_scope": 3,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def normalize_arabic(text: str) -> str:
    value = str(text).strip().lower()
    value = re.sub(r"[إأآٱ]", "ا", value)
    value = (
        value.replace("ى", "ي")
        .replace("ة", "ه")
        .replace("ؤ", "و")
        .replace("ئ", "ي")
    )
    value = re.sub(r"[\u064b-\u065f\u0670\u06d6-\u06ed]", "", value)
    value = re.sub(r"[^\u0600-\u06ff0-9a-z]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def gold_articles(questions: list[dict]) -> set[int]:
    values: set[int] = set()
    for case in questions:
        if case.get("expected_behavior", "retrieve") != "retrieve":
            continue
        for key in ("required_articles", "acceptable_articles"):
            values.update(int(v) for v in case.get(key, []))
        if "primary_article" in case:
            values.add(int(case["primary_article"]))
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--ttl", type=Path, default=DEFAULT_TTL)
    parser.add_argument("--dev20", type=Path, default=DEFAULT_DEV20)
    parser.add_argument("--previous50", type=Path, default=DEFAULT_PREVIOUS50)
    parser.add_argument("--previous30", type=Path, default=DEFAULT_PREVIOUS30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    benchmark = load_json(args.benchmark)
    questions = benchmark["questions"]

    assert benchmark.get("frozen") is True
    assert len(questions) == 20
    assert len({case["id"] for case in questions}) == 20

    normalized = [normalize_arabic(case["question"]) for case in questions]
    assert len(set(normalized)) == 20

    distribution = Counter(case["test_type"] for case in questions)
    assert dict(distribution) == EXPECTED_DISTRIBUTION
    assert benchmark["distribution"] == EXPECTED_DISTRIBUTION

    behavior_counts = Counter(
        case["expected_behavior"] for case in questions
    )
    assert behavior_counts == {
        "retrieve": 14,
        "clarify": 3,
        "abstain": 3,
    }

    graph = Graph()
    graph.parse(args.ttl)
    namespace = Namespace("http://example.org/jordan-labor-law#")

    concept_names = {
        str(subject).split("#")[-1]
        for subject in graph.subjects(RDFS.label, None)
        if str(subject).startswith(str(namespace))
    }

    for case in questions:
        behavior = case["expected_behavior"]
        if behavior == "retrieve":
            required = [int(v) for v in case["required_articles"]]
            acceptable = [
                int(v) for v in case.get("acceptable_articles", [])
            ]
            assert int(case["primary_article"]) in required
            assert required
            for number in set(required + acceptable):
                article_uri = namespace[f"article_{number}"]
                assert any(
                    graph.objects(article_uri, RDFS.comment)
                ), (case["id"], number)
            for concept in case.get("expected_concepts_any", []):
                assert concept in concept_names, (case["id"], concept)
        else:
            assert "primary_article" not in case
            assert "required_articles" not in case
            assert "acceptable_articles" not in case

    prior_paths = [args.dev20, args.previous50, args.previous30]
    prior_questions: list[str] = []
    prior_gold: set[int] = set()

    for path in prior_paths:
        prior = load_json(path)
        prior_questions.extend(
            str(case["question"]) for case in prior["questions"]
        )
        prior_gold.update(gold_articles(prior["questions"]))

    current_gold = gold_articles(questions)
    overlap = current_gold & prior_gold
    assert not overlap, f"Gold article overlap: {sorted(overlap)}"

    prior_normalized = {
        normalize_arabic(question) for question in prior_questions
    }
    exact_question_overlap = set(normalized) & prior_normalized
    assert not exact_question_overlap, exact_question_overlap

    highest_similarity = 0.0
    highest_pair: tuple[str, str] | None = None
    for case in questions:
        current = normalize_arabic(case["question"])
        for previous in prior_questions:
            score = SequenceMatcher(
                None, current, normalize_arabic(previous)
            ).ratio()
            if score > highest_similarity:
                highest_similarity = score
                highest_pair = (case["question"], previous)

    assert highest_similarity < 0.70, (
        highest_similarity,
        highest_pair,
    )

    print("Stage 7.7-D unseen 20 holdout validation passed.")
    print("Questions: 20")
    print("Retrieve / clarify / abstain: 14 / 3 / 3")
    print(f"Distribution: {dict(distribution)}")
    print(f"New gold articles: {sorted(current_gold)}")
    print(f"Prior gold overlap: {sorted(overlap)}")
    print(f"Highest prior-question similarity: {highest_similarity:.4f}")
    print(f"SHA-256: {sha256(args.benchmark)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
