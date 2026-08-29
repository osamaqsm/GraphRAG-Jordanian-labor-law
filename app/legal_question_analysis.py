from __future__ import annotations

import re
import unicodedata


_ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_NON_WORD_RE = re.compile(r"[^0-9A-Za-z\u0600-\u06FF]+")

_ARABIC_STOPWORDS = {
    "انا", "ان", "إن", "او", "أو", "اي", "أي", "اذا", "إذا", "الى", "إلى",
    "التي", "الذي", "على", "عن", "في", "ما", "ماذا", "من", "هل", "هو",
    "هي", "هذا", "هذه", "كم", "كيف", "له", "لها", "لم", "لا", "مع", "بعد",
    "قبل", "يمكن", "يجب", "وفق", "ضمن", "احكام", "أحكام", "قانون", "القانون",
    "العمل", "الاردني", "الأردني", "اردني", "أردني",
}


def normalize_arabic(text: str) -> str:
    """Normalize Arabic for lexical matching without changing legal content."""

    value = unicodedata.normalize("NFKC", str(text or ""))
    value = _ARABIC_DIACRITICS_RE.sub("", value)
    value = value.translate(
        str.maketrans(
            {
                "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
                "ى": "ي", "ؤ": "و", "ئ": "ي", "ـ": "",
                "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
                "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
                "۰": "0", "۱": "1", "۲": "2", "۳": "3", "۴": "4",
                "۵": "5", "۶": "6", "۷": "7", "۸": "8", "۹": "9",
            }
        )
    )
    value = _NON_WORD_RE.sub(" ", value.lower())
    return " ".join(value.split())


def content_tokens(text: str) -> tuple[str, ...]:
    normalized = normalize_arabic(text)
    tokens = [
        token
        for token in normalized.split()
        if len(token) > 1 and token not in _ARABIC_STOPWORDS
    ]
    return tuple(dict.fromkeys(tokens))


def article_question_relevance(question: str, article_text: str) -> float:
    """Deterministic issue relevance used only after graph retrieval.

    This score never discovers a new Article node. It only reorders articles
    that have already been reached through a valid graph path.
    """

    q_tokens = set(content_tokens(question))
    if not q_tokens:
        return 0.0

    a_normalized = normalize_arabic(article_text)
    a_tokens = set(a_normalized.split())

    overlap = len(q_tokens & a_tokens) / len(q_tokens)

    q_numbers = {token for token in q_tokens if token.isdigit()}
    number_score = 0.0
    if q_numbers:
        number_score = len(q_numbers & a_tokens) / len(q_numbers)

    phrase = normalize_arabic(question)
    phrase_bonus = 1.0 if phrase and phrase in a_normalized else 0.0

    score = 0.75 * overlap + 0.20 * number_score + 0.05 * phrase_bonus
    return max(0.0, min(1.0, score))
