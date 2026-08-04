from __future__ import annotations

from pydantic import ValidationError

from app.legal_article_reranker import ArticleSelection
from app.legal_question_analysis import analyze_legal_question


def main() -> None:
    accompaniment = analyze_legal_question(
        "انتقل زوجي للعمل خارج المحافظة؛ هل أستطيع أخذ إجازة لمرافقته وكم أقصى مدتها؟"
    )
    assert accompaniment.behavior == "retrieve", accompaniment

    ambiguous_leave = analyze_legal_question(
        "كم مدة الإجازة التي أستحقها؟"
    )
    assert ambiguous_leave.behavior == "clarify", ambiguous_leave

    traffic = analyze_legal_question(
        "وصلتني مخالفة سير بسبب تجاوز السرعة، كيف أعترض عليها؟"
    )
    assert traffic.behavior == "abstain", traffic

    multi = ArticleSelection(
        behavior="retrieve",
        selected_article_numbers=[39, 40],
        confidence=0.94,
        reason="Two independent collective-contract rules are requested.",
        clarification_question="",
    )
    assert multi.selected_article_numbers == [39, 40]

    try:
        ArticleSelection(
            behavior="retrieve",
            selected_article_numbers=[1, 2, 3, 4],
            confidence=0.9,
            reason="Too many articles.",
            clarification_question="",
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("ArticleSelection accepted more than 3 articles.")

    print("Stage 7.6-B offline checks passed.")
    print("Accompaniment leave route: retrieve")
    print("Ambiguous leave route: clarify")
    print("Traffic route: abstain")
    print("Constrained multi-article schema: [39, 40]")


if __name__ == "__main__":
    main()
