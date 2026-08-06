import json
import requests

CASES = [
    # يجب أن تكون retrieve
    (
        "IN-01",
        "إذا أصيب العامل أثناء العمل، متى يستحق تعويضاً إضافياً؟",
        "retrieve",
    ),
    (
        "IN-02",
        "من يحدد مكافآت رئيس وأعضاء المحكمة العمالية؟",
        "retrieve",
    ),
    (
        "IN-03",
        "ما عقوبة النقابة إذا خالفت نظامها الداخلي، وكيف يتم تصويب المخالفة؟",
        "retrieve",
    ),
    (
        "IN-04",
        "هل تنشأ مسؤولية مدنية أو جزائية على عضو النقابة بسبب أعماله النقابية؟",
        "retrieve",
    ),
    (
        "IN-05",
        "هل يجوز تنظيم العمال في نقابات رغم ارتباط ذلك بحرية التجارة؟",
        "retrieve",
    ),
    (
        "IN-06",
        "متى تكتسب النقابة الشخصية الاعتبارية؟",
        "retrieve",
    ),
    (
        "IN-07",
        "ما وضع النقابات القائمة قبل نفاذ قانون العمل الحالي؟",
        "retrieve",
    ),

    # يجب أن تكون abstain
    (
        "OOS-01",
        "ما شروط استحقاق راتب التقاعد المبكر من الضمان الاجتماعي؟",
        "abstain",
    ),
    (
        "OOS-02",
        "كيف يُحتسب راتب الاعتلال وبدل التعطل من مؤسسة الضمان؟",
        "abstain",
    ),
    (
        "OOS-03",
        "هل يجوز للبنك اقتطاع أقساط القرض والفوائد من حساب الراتب؟",
        "abstain",
    ),
    (
        "OOS-04",
        "ما إجراءات تجديد إقامة عامل أجنبي وتأشيرته ولم شمل أسرته؟",
        "abstain",
    ),
]

passed = 0

for case_id, question, expected in CASES:
    response = requests.post(
        "http://localhost:8000/retrieve",
        json={
            "question": question,
            "include_debug": False,
        },
        timeout=180,
    )
    response.raise_for_status()
    result = response.json()

    actual = result["decision"]["behavior"]
    articles = result.get("diagnostics", {}).get("article_numbers", [])
    ok = actual == expected
    passed += int(ok)

    print(
        json.dumps(
            {
                "id": case_id,
                "expected": expected,
                "actual": actual,
                "passed": ok,
                "confidence": result["decision"].get("planner_confidence"),
                "reason": result["decision"].get("reason"),
                "articles": articles,
            },
            ensure_ascii=False,
        )
    )

print()
print(f"Routing result: {passed}/{len(CASES)}")
