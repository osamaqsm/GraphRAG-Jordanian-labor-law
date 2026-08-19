from __future__ import annotations

import json

import requests

from app.config import get_settings


QUESTION = (
    "ما القاعدة العامة بشأن الإجازة السنوية للعامل في قانون العمل الأردني؟"
)


def main() -> int:
    settings = get_settings()
    print(
        json.dumps(
            {
                "provider": settings.pipeline_llm_provider,
                "model": settings.pipeline_llm_model,
                "strict": True,
                "planner_max_output_tokens": settings.planner_max_output_tokens,
                "reranker_max_output_tokens": settings.reranker_max_output_tokens,
                "generator_max_output_tokens": settings.generator_max_output_tokens,
                "reranker_candidate_limit": settings.reranker_candidate_limit,
                "reranker_total_char_budget": settings.reranker_total_char_budget,
                "ollama_num_ctx": settings.ollama_num_ctx,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    retrieval_response = requests.post(
        "http://localhost:8000/retrieve",
        json={"question": QUESTION, "include_debug": False},
        timeout=600,
    )
    print("retrieve_status:", retrieval_response.status_code)
    print("retrieve_body:", retrieval_response.text[:3000])
    retrieval_response.raise_for_status()
    retrieval = retrieval_response.json()

    generation_response = requests.post(
        "http://localhost:8000/generate?include_debug=false",
        json=retrieval,
        timeout=600,
    )
    print("generate_status:", generation_response.status_code)
    print("generate_body:", generation_response.text[:3000])
    generation_response.raise_for_status()

    generation = generation_response.json()
    planner_used = retrieval.get("decision", {}).get("planner_used")
    generated_model = generation.get("model")

    if planner_used is not True:
        raise RuntimeError("Smoke test failed: planner_used is not true.")
    if generation.get("status") == "generated" and generated_model != settings.pipeline_llm_model:
        raise RuntimeError(
            f"Smoke test failed: generated model={generated_model!r}, "
            f"expected {settings.pipeline_llm_model!r}."
        )

    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
