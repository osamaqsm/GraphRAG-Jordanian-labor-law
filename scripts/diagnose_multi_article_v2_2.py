from __future__ import annotations

import json

from app.retrieval_pipeline import RetrievalOnlyPipeline

BENCHMARK = "/app/data/benchmarks/jordan_labor_law_final_unseen_40.json"

GOLD = {
    "FM40-30": [12, 13, 14],
    "FM40-31": [21, 25, 26, 27, 31],
    "FM40-32": [39, 40, 41, 42, 43],
    "FM40-33": [57, 59, 60, 61],
    "FM40-34": [67, 68, 70, 71, 72],
    "FM40-35": [86, 87, 88, 93, 94],
    "FM40-36": [120, 121, 122, 124, 126],
}


def main() -> None:
    with open(BENCHMARK, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    by_id = {item["id"]: item for item in data["questions"]}
    recalls: list[float] = []

    with RetrievalOnlyPipeline() as pipeline:
        for qid, gold in GOLD.items():
            item = by_id[qid]
            result = pipeline.retrieve(item["question"], include_debug=True)
            payload = result.model_dump(mode="json")
            retrieved = payload["diagnostics"]["article_numbers"]
            debug = payload.get("debug") or {}
            issue_debug = debug.get("issue_wise_retrieval") or {}
            coverage = issue_debug.get("coverage") or {}

            gold_set = set(gold)
            retrieved_set = set(retrieved)
            matched = gold_set & retrieved_set
            recall = len(matched) / len(gold_set)
            precision = (
                len(matched) / len(retrieved_set)
                if retrieved_set
                else 0.0
            )
            recalls.append(recall)

            print("=" * 80)
            print(qid)
            print("GOLD:      ", gold)
            print("RETRIEVED: ", retrieved)
            print("MATCHED:   ", sorted(matched))
            print(f"RECALL:     {recall:.3f}")
            print(f"PRECISION:  {precision:.3f}")
            print("ISSUE CANDIDATES:", coverage.get("issue_candidate_numbers"))
            print("STRUCTURAL NEIGHBORS:", coverage.get("structural_neighbor_numbers"))
            print("RERANKER:", coverage.get("reranker_selected_numbers"))
            print("COVERAGE:")
            for row in coverage.get("coverage", []):
                print(
                    " ",
                    row.get("issue_index"),
                    row.get("issue_ar"),
                    "=>",
                    row.get("support_articles", [row.get("support_article")]),
                    "|",
                    row.get("source"),
                )

    print("=" * 80)
    print("MULTI-ARTICLE SUMMARY")
    print(f"Mean Recall: {sum(recalls) / len(recalls):.3f}")
    print("Perfect cases:", sum(value == 1.0 for value in recalls), "/", len(recalls))


if __name__ == "__main__":
    main()
