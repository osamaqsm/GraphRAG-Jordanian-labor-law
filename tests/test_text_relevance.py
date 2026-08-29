from app.legal_question_analysis import article_question_relevance, normalize_arabic


def test_arabic_normalization_unifies_alef_and_digits():
    assert normalize_arabic("إجازة ٦٠ يوماً") == "اجازة 60 يوما"


def test_relevance_prefers_matching_statutory_text():
    question = "بعد كم يوما يبدأ نفاذ القانون بعد نشره في الجريدة الرسمية"
    good = "يعمل به بعد مرور ستين يوما على تاريخ نشره في الجريدة الرسمية"
    bad = "للعامل الحق في الاجازة السنوية"
    assert article_question_relevance(question, good) > article_question_relevance(question, bad)
