from __future__ import annotations

from app.legal_question_analysis import analyze_legal_question


CASES = (
    ("كم مدة الإجازة التي أستحقها؟", "clarify"),
    ("هل يقدر صاحب العمل يفصلني؟", "clarify"),
    ("شو نسبة التعويض إلي؟", "clarify"),
    ("صاحب العمل حسم من راتبي، هل الحسم قانوني؟", "clarify"),
    ("عندي عقد عمل، شو حقوقي؟", "clarify"),
    ("وصلتني مخالفة سير بسبب تجاوز السرعة، كيف أعترض عليها؟", "abstain"),
    ("المالك يريد رفع أجرة الشقة قبل انتهاء عقد الإيجار، ما حقوقي؟", "abstain"),
    ("بعد الطلاق، كيف تحدد حضانة الأطفال والنفقة؟", "abstain"),
    ("كيف أحسب ضريبة الدخل السنوية على نشاطي التجاري؟", "abstain"),
    ("تعرضت لاعتداء في الشارع من شخص لا علاقة له بعملي، ما العقوبة الجنائية؟", "abstain"),
    ("كم مدة الإجازة المرضية المدفوعة؟", "retrieve"),
    ("فصلني صاحب العمل تعسفياً، هل أستحق تعويضاً؟", "retrieve"),
    ("اقتطع صاحب العمل من أجري بسبب سلفة، ما الحد المسموح؟", "retrieve"),
)


def main() -> int:
    for question, expected in CASES:
        analysis = analyze_legal_question(question)
        print(
            f"{expected:8s} | actual={analysis.behavior:8s} | "
            f"max_articles={analysis.max_final_articles} | {question}"
        )
        if analysis.behavior != expected:
            raise AssertionError(
                f"Expected {expected!r}, got {analysis.behavior!r}: {question}"
            )

    print("Stage 7.6-A routing checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
