from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


_ARABIC_DIACRITICS = re.compile(
    r"[\u0617-\u061A\u064B-\u0652\u0670\u06D6-\u06ED]"
)
_NON_WORD = re.compile(r"[^0-9\u0621-\u064A]+")
_SPACE = re.compile(r"\s+")

_ARABIC_DIGITS = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)

# Longer expressions must be replaced first.
_NUMBER_EXPRESSIONS: tuple[tuple[str, str], ...] = (
    ("مئه وخمسين", "150"),
    ("مائه وخمسين", "150"),
    ("مئه وخمسه وعشرين", "125"),
    ("مائه وخمسه وعشرين", "125"),
    ("واحد وعشرين", "21"),
    ("احد وعشرين", "21"),
    ("اربعه عشر", "14"),
    ("احد عشر", "11"),
    ("عشره", "10"),
    ("ثلاثين", "30"),
    ("عشرين", "20"),
    ("خمسه", "5"),
    ("سته", "6"),
)

_STOPWORDS = {
    "انا",
    "ان",
    "او",
    "اي",
    "اذا",
    "الى",
    "التي",
    "الذي",
    "الذين",
    "على",
    "عن",
    "في",
    "ما",
    "ماذا",
    "من",
    "منذ",
    "هل",
    "هو",
    "هي",
    "هذا",
    "هذه",
    "كان",
    "كانت",
    "كم",
    "كيف",
    "له",
    "لها",
    "لم",
    "لا",
    "مع",
    "بعد",
    "قبل",
    "لدى",
    "نفسه",
    "يمكن",
    "يجب",
}


@dataclass(frozen=True, slots=True)
class IssueProfile:
    issue_id: str
    trigger_groups: tuple[tuple[str, ...], ...]
    preferred_concepts: tuple[str, ...]
    query_expansion_terms: tuple[str, ...]
    article_anchor_phrases: tuple[str, ...]
    primary_article_anchor_phrases: tuple[str, ...] = ()
    exclude_phrases: tuple[str, ...] = ()
    max_final_articles: int = 1


@dataclass(frozen=True, slots=True)
class LegalQuestionAnalysis:
    original_question: str
    normalized_question: str
    issue_ids: tuple[str, ...]
    preferred_concepts: tuple[str, ...]
    query_expansion_terms: tuple[str, ...]
    article_anchor_phrases: tuple[str, ...]
    primary_article_anchor_phrases: tuple[str, ...]
    meaningful_tokens: frozenset[str]
    numeric_tokens: frozenset[str]
    max_final_articles: int
    behavior: str
    behavior_reason: str

    @property
    def bm25_query(self) -> str:
        additions = [
            term
            for term in self.query_expansion_terms
            if normalize_arabic(term) not in self.normalized_question
        ]

        if not additions:
            return self.original_question

        return " ".join(
            [self.original_question, *additions]
        )

    @property
    def anchor_query(self) -> str:
        """Issue-specific legal phrases used for high-recall BM25."""

        if not self.article_anchor_phrases:
            return ""

        return " ".join(
            self.article_anchor_phrases
        )

    @property
    def anchor_queries(self) -> tuple[str, ...]:
        """
        Return a small set of focused BM25 queries.

        A single long OR-style BM25 query can be dominated by generic
        legal words.  Pairing related statutory anchors preserves recall
        while rewarding an article that matches several distinct aspects
        of the legal issue.
        """

        if not self.article_anchor_phrases:
            return ()

        queries: list[str] = [self.anchor_query]
        phrases = list(self.article_anchor_phrases)

        for index in range(0, len(phrases), 2):
            query = " ".join(phrases[index:index + 2])
            if query and query not in queries:
                queries.append(query)

        return tuple(queries)


ISSUE_PROFILES: tuple[IssueProfile, ...] = (
    IssueProfile(
        issue_id="wage_delay",
        trigger_groups=(
            ("راتب", "اجر", "اجور"),
            ("لم يدفع", "تاخر", "متاخر", "تجاوز"),
        ),
        preferred_concepts=(
            "delay_exceeds_seven_days_condition",
            "wage_delay_violation",
            "pay_wages_obligation",
        ),
        query_expansion_terms=(
            "تاخير دفع الاجر",
            "الاجر المتاخر",
        ),
        article_anchor_phrases=(
            "دفع الاجر",
            "خلال مده لا تزيد على سبعه ايام",
            "اجور متاخره",
            "سلطه الاجور",
        ),
        primary_article_anchor_phrases=(
            "دفع الاجر",
            "خلال مده لا تزيد على سبعه ايام",
        ),
        max_final_articles=2,
    ),
    IssueProfile(
        issue_id="oral_contract_proof",
        trigger_groups=(
            ("عقد شفهي", "عقد شفوي", "عقد غير مكتوب", "لم يكتب", "على الورق"),
            ("اثبات", "حقوقي", "حقوق العامل"),
        ),
        preferred_concepts=(
            "oral_contract_concept",
            "written_contract_concept",
            "employment_contract_concept",
        ),
        query_expansion_terms=(
            "عقد العمل الشفهي",
            "اثبات الحقوق",
            "لم يحرر العقد كتابه",
        ),
        article_anchor_phrases=(
            "يجوز للعامل اثبات حقوقه",
            "جميع طرق الاثبات القانونيه",
            "اذا لم يحرر العقد كتابه",
            "ينظم عقد العمل باللغه العربيه",
        ),
    ),
    IssueProfile(
        issue_id="fixed_term_early_termination",
        trigger_groups=(
            ("محدد المده", "مده محدوده"),
            ("قبل انتهاء", "انهاه", "انهاء العقد"),
        ),
        preferred_concepts=(
            "contract_fixed_term_condition",
            "fixed_term_contract_concept",
            "termination_contract_consequence",
        ),
        query_expansion_terms=(
            "انهاء عقد محدد المده قبل انتهاء مدته",
            "الاجور المستحقه حتى نهايه المده",
        ),
        article_anchor_phrases=(
            "عقد العمل المحدد المده",
            "قبل انتهاء مدته",
            "الحقوق والمزايا",
            "الاجور التي تستحق حتى انتهاء المده",
        ),
        exclude_phrases=(
            "غير محدد المده",
            "مده غير محدوده",
        ),
    ),
    IssueProfile(
        issue_id="indefinite_contract_notice",
        trigger_groups=(
            ("غير محدد المده", "مده غير محدوده"),
            ("اشعار", "انهاء العقد"),
        ),
        preferred_concepts=(
            "contract_indefinite_term_condition",
            "indefinite_term_contract_concept",
            "dismissal_event",
        ),
        query_expansion_terms=(
            "اشعار خطي قبل انهاء عقد غير محدد المده",
        ),
        article_anchor_phrases=(
            "عقد العمل غير المحدد المده",
            "اشعار الطرف الاخر خطيا",
            "قبل شهر واحد على الاقل",
        ),
    ),
    IssueProfile(
        issue_id="complaint_retaliation",
        trigger_groups=(
            ("شكوى", "شكوي", "شكاوى", "مطالبه", "وزارة العمل", "الجهات المختصه"),
            ("فصل", "عاقب", "تاديبي", "اجراء"),
        ),
        preferred_concepts=(
            "arbitrary_dismissal_violation",
            "dismissal_event",
            "disciplinary_action_consequence",
        ),
        query_expansion_terms=(
            "الشكاوى والمطالبات",
            "الشكاوي والمطالبات",
            "اجراء تاديبي بسبب الشكوى",
            "فصل العامل بسبب الشكوى",
            "الجهات المختصه",
        ),
        article_anchor_phrases=(
            "لا يجوز فصل العامل",
            "اتخاذ اي اجراء تاديبي",
            "لاسباب تتصل بالشكاوي والمطالبات",
            "الجهات المختصه",
        ),
    ),
    IssueProfile(
        issue_id="arbitrary_dismissal",
        trigger_groups=(
            ("فصل تعسفي", "تعسفي", "تعسفيا", "دون سبب", "بدون سبب", "فصل غير مشروع"),
            ("تعويض", "اعاده للعمل", "فصلني"),
        ),
        preferred_concepts=(
            "arbitrary_dismissal_violation",
            "dismissal_without_lawful_ground_condition",
            "dismissal_event",
        ),
        query_expansion_terms=(
            "الفصل التعسفي",
            "دعوى الفصل",
            "التعويض عن الفصل",
        ),
        article_anchor_phrases=(
            "الفصل كان تعسفيا",
            "خلال ستين يوما",
            "اعاده العامل الى عمله",
            "بدفع تعويض",
        ),
    ),
    IssueProfile(
        issue_id="absence_dismissal",
        trigger_groups=(
            ("تغيب", "غياب", "تغيبت"),
            ("متتالي", "متتاليه", "دون عذر", "دون سبب مشروع", "ايام"),
        ),
        preferred_concepts=(
            "work_negligence_violation",
            "dismissal_event",
            "termination_contract_consequence",
        ),
        query_expansion_terms=(
            "تغيب العامل دون سبب مشروع",
            "اكثر من عشره ايام متتاليه",
            "الفصل دون اشعار",
        ),
        article_anchor_phrases=(
            "تغيب العامل دون سبب مشروع",
            "اكثر من عشره ايام متتاليه",
            "الفصل دون اشعار",
            "انذار كتابي",
        ),
    ),
    IssueProfile(
        issue_id="different_work_assignment",
        trigger_groups=(
            ("عمل مختلف", "يختلف اختلافا", "غير العمل المتفق", "كلفني"),
            ("العمل المتفق عليه", "ترك العمل", "يلزمني"),
        ),
        preferred_concepts=(
            "assignment_different_work_violation",
            "assign_agreed_work_obligation",
        ),
        query_expansion_terms=(
            "عمل يختلف اختلافا بينا",
            "العمل المتفق عليه",
        ),
        article_anchor_phrases=(
            "عمل يختلف اختلافا بينا",
            "طبيعه العمل المتفق عليه",
            "ترك العمل دون اشعار",
            "الاحتفاظ بحقوقه القانونيه",
        ),
        primary_article_anchor_phrases=(
            "عمل يختلف اختلافا بينا",
            "طبيعه العمل المتفق عليه",
        ),
        max_final_articles=2,
    ),
    IssueProfile(
        issue_id="confidentiality",
        trigger_groups=(
            ("اسرار", "سريه", "افشاء"),
            ("صاحب العمل", "العمل"),
        ),
        preferred_concepts=(
            "disclosure_employer_secrets_violation",
            "preserve_confidentiality_obligation",
        ),
        query_expansion_terms=(
            "المحافظه على اسرار صاحب العمل",
            "افشاء اسرار العمل",
        ),
        article_anchor_phrases=(
            "المحافظه على اسرار صاحب العمل",
            "لا يفشيها",
            "افشى العامل الاسرار الخاصه بالعمل",
            "فصل العامل دون اشعار",
        ),
        primary_article_anchor_phrases=(
            "المحافظه على اسرار صاحب العمل",
            "لا يفشيها",
        ),
        max_final_articles=2,
    ),
    IssueProfile(
        issue_id="annual_leave",
        trigger_groups=(
            ("اجازه سنويه", "الاجازه السنويه"),
        ),
        preferred_concepts=(
            "leave_right_concept",
            "leave_request_event",
            "wage_right_concept",
        ),
        query_expansion_terms=(
            "الاجازه السنويه باجر كامل",
            "سنوات الخدمه لدى صاحب العمل نفسه",
        ),
        article_anchor_phrases=(
            "اجازه سنويه باجر كامل",
            "خمس سنوات متصله",
            "واحدا وعشرين يوما",
            "اربع عشر يوما عن كل سنه خدمه",
        ),
    ),
    IssueProfile(
        issue_id="sick_leave",
        trigger_groups=(
            ("اجازه مرضيه", "الاجازه المرضيه"),
        ),
        preferred_concepts=(
            "sick_leave_right_concept",
            "leave_right_concept",
        ),
        query_expansion_terms=(
            "الاجازه المرضيه باجر كامل",
            "تجديد الاجازه المرضيه",
        ),
        article_anchor_phrases=(
            "اجازه مرضيه",
            "اربعه عشر يوما باجر كامل",
            "تجدد لمده اربعه عشر يوما اخرى",
            "تقرير من اللجنه الطبيه",
        ),
    ),
    IssueProfile(
        issue_id="maternity_leave",
        trigger_groups=(
            ("اجازه امومه", "الامومه"),
            ("ولاده", "الوضع", "بعد الولاده", "بعد الوضع", "مده"),
        ),
        preferred_concepts=(
            "leave_right_concept",
            "leave_request_event",
            "wage_right_concept",
        ),
        query_expansion_terms=(
            "اجازه الامومه باجر كامل",
            "قبل الوضع وبعده",
        ),
        article_anchor_phrases=(
            "اجازه امومه باجر كامل",
            "قبل الوضع وبعده",
            "مجموع مدتها عشره اسابيع",
            "بعد الوضع عن سته اسابيع",
        ),
    ),
    IssueProfile(
        issue_id="end_of_service",
        trigger_groups=(
            ("مكافاه نهايه الخدمه", "نهايه الخدمه"),
            ("ضمان اجتماعي", "غير مشمول", "غير خاضع"),
        ),
        preferred_concepts=(
            "worker_not_covered_social_security_condition",
            "end_of_service_benefit_right_concept",
            "end_of_service_benefit_consequence",
        ),
        query_expansion_terms=(
            "العامل غير الخاضع للضمان الاجتماعي",
            "مكافاه نهايه الخدمه",
        ),
        article_anchor_phrases=(
            "غير الخاضع لاحكام قانون الضمان الاجتماعي",
            "مكافاه نهايه الخدمه",
            "اجر شهر عن كل سنه",
            "كسور السنه",
        ),
    ),
    IssueProfile(
        issue_id="overtime_pay",
        trigger_groups=(
            ("عمل اضافي", "العمل الاضافي", "ساعه اضافيه", "ساعات اضافيه"),
        ),
        preferred_concepts=(
            "wage_right_concept",
            "wage_payment_event",
            "pay_wages_obligation",
        ),
        query_expansion_terms=(
            "ساعه العمل الاضافيه",
            "اجر اضافي",
            "العمل في يوم العطله الاسبوعيه",
        ),
        article_anchor_phrases=(
            "ساعه العمل الاضافيه",
            "125 من اجره المعتاد",
            "اشتغل العامل في يوم عطلته الاسبوعيه",
            "150 من اجره المعتاد",
        ),
    ),
    IssueProfile(
        issue_id="weekly_rest",
        trigger_groups=(
            ("العطله الاسبوعيه", "يوم العطله", "يوم الجمعه"),
            ("اجر كامل", "باجر كامل", "ما يوم", "اي يوم"),
        ),
        preferred_concepts=(
            "weekly_rest_right_concept",
            "wage_right_concept",
        ),
        query_expansion_terms=(
            "يوم العطله الاسبوعيه",
            "العطله الاسبوعيه باجر كامل",
        ),
        article_anchor_phrases=(
            "يوم الجمعه من كل اسبوع",
            "يوم العطله الاسبوعيه",
            "باجر كامل",
            "عمل سته ايام متصله",
        ),
    ),
    IssueProfile(
        issue_id="minimum_wage",
        trigger_groups=(
            ("الحد الادنى", "اقل من الحد الادنى"),
            ("اجر", "راتب", "فرق الاجر"),
        ),
        preferred_concepts=(
            "wage_right_concept",
            "wage_payment_event",
            "pay_wages_obligation",
        ),
        query_expansion_terms=(
            "اجر اقل من الحد الادنى",
            "فرق الاجر",
            "غرامه صاحب العمل",
        ),
        article_anchor_phrases=(
            "اجرا يقل عن الحد الادنى",
            "فرق الاجر",
            "غرامه لا تقل",
            "عن كل حاله يدفع فيها",
        ),
    ),
    IssueProfile(
        issue_id="occupational_safety",
        trigger_groups=(
            ("معدات الوقايه", "وسائل الوقايه", "الحمايه الشخصيه", "مخاطر المهنه"),
            ("صاحب العمل", "تكلفتها", "العامل"),
        ),
        preferred_concepts=(
            "provide_safe_work_environment_obligation",
            "safe_work_environment_right_concept",
        ),
        query_expansion_terms=(
            "توفير معدات الوقايه الشخصيه",
            "تعريف العامل بمخاطر المهنه",
            "عدم تحميل العامل تكلفتها",
        ),
        article_anchor_phrases=(
            "الاحتياطات والتدابير اللازمه",
            "وسائل واجهزه الوقايه الشخصيه",
            "تعريف العامل بمخاطر مهنته",
            "لا يجوز تحميل العمال اي نفقات",
        ),
    ),
    IssueProfile(
        issue_id="juvenile_working_hours",
        trigger_groups=(
            ("عامل حدث", "الحدث", "قاصر"),
            ("ست ساعات", "ساعات", "الثامنه مساء", "السادسه صباحا", "ليلا"),
        ),
        preferred_concepts=(
            "juvenile_concept",
            "weekly_rest_right_concept",
        ),
        query_expansion_terms=(
            "تشغيل الحدث اكثر من ست ساعات",
            "بين الثامنه مساء والسادسه صباحا",
        ),
        article_anchor_phrases=(
            "يحظر تشغيل الحدث",
            "اكثر من ست ساعات",
            "بين الساعه الثامنه مساء والسادسه صباحا",
            "ايام العطله الاسبوعيه",
        ),
    ),
    IssueProfile(
        issue_id="work_injury_notice",
        trigger_groups=(
            ("اصابه عمل", "اصيب عامل", "اصابه العامل"),
            ("ابلاغ", "اشعار", "الوزاره", "كم ساعه", "خلال"),
        ),
        preferred_concepts=(
            "work_injury_event",
            "employer_liability_consequence",
            "ministry_concept",
        ),
        query_expansion_terms=(
            "اشعار الوزاره باصابه العمل",
            "خلال ثمان واربعين ساعه",
        ),
        article_anchor_phrases=(
            "اشعارا الى الوزاره",
            "خلال مده لا تزيد على 48 ساعه",
            "اصابه عمل",
            "نقل المصاب",
        ),
    ),
    IssueProfile(
        issue_id="total_disability_compensation",
        trigger_groups=(
            ("اصابه العمل", "اصابه عمل"),
            ("عجز كلي", "العجز الكلي", "تعويض"),
        ),
        preferred_concepts=(
            "work_injury_event",
            "compensation_consequence",
            "medical_examination_event",
        ),
        query_expansion_terms=(
            "تعويض العجز الكلي",
            "اصابه العمل",
        ),
        article_anchor_phrases=(
            "العجز الكلي",
            "اجر الف ومئتي يوم عمل",
            "التعويض الواجب دفعه",
            "اصابه العمل",
        ),
    ),
)


def normalize_arabic(text: str) -> str:
    normalized = str(text or "").translate(
        _ARABIC_DIGITS
    )
    normalized = _ARABIC_DIACRITICS.sub(
        "",
        normalized,
    )
    normalized = normalized.replace("ـ", "")
    normalized = normalized.translate(
        str.maketrans(
            {
                "أ": "ا",
                "إ": "ا",
                "آ": "ا",
                "ٱ": "ا",
                "ؤ": "و",
                "ئ": "ي",
                "ى": "ي",
                "ة": "ه",
            }
        )
    )
    normalized = _NON_WORD.sub(
        " ",
        normalized,
    )
    normalized = _SPACE.sub(
        " ",
        normalized,
    ).strip()

    # Apply normalized Arabic number expressions after letter normalization.
    padded = f" {normalized} "
    for expression, replacement in _NUMBER_EXPRESSIONS:
        padded = padded.replace(
            f" {expression} ",
            f" {replacement} ",
        )

    return _SPACE.sub(
        " ",
        padded,
    ).strip()


def _profile_matches(
    normalized_question: str,
    profile: IssueProfile,
) -> bool:
    if any(
        normalize_arabic(excluded)
        in normalized_question
        for excluded in profile.exclude_phrases
    ):
        return False

    return all(
        any(
            normalize_arabic(trigger)
            in normalized_question
            for trigger in group
        )
        for group in profile.trigger_groups
    )


def _deduplicate(items: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []

    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)

    return tuple(result)



_OUT_OF_SCOPE_GROUPS: tuple[tuple[str, ...], ...] = (
    (
        "مخالفه سير",
        "تجاوز السرعه",
        "رخصه قياده",
        "اعتراض على مخالفه",
    ),
    (
        "عقد الايجار",
        "اجره الشقه",
        "ايجار الشقه",
        "المستاجر",
        "رفع الايجار",
    ),
    (
        "بعد الطلاق",
        "حضانه الاطفال",
        "الحضانه والنفقه",
        "قانون الاحوال الشخصيه",
    ),
    (
        "ضريبه الدخل",
        "الاقرار الضريبي",
        "نشاطي التجاري",
        "الضريبه السنويه",
    ),
    (
        "عقوبه جنائيه",
        "اعتداء في الشارع",
        "جريمه في الشارع",
        "لا علاقه له بعملي",
    ),
)

_LEAVE_SPECIFIERS = (
    "سنويه",
    "مرضيه",
    "امومه",
    "ولاده",
    "رضاعه",
    "حج",
    "دراسه",
    "تربيه اطفال",
    "مرافقه",
    "مرافق",
    "لمرافق",
    "دون راتب",
    "دون اجر",
)

_DISMISSAL_DETAIL_MARKERS = (
    "تعسفي",
    "شكوى",
    "شكوي",
    "غياب",
    "تغيب",
    "اسرار",
    "سر",
    "انذار",
    "اشعار",
    "اعتداء",
    "سكر",
    "مخدر",
    "خطا",
    "شكوى",
    "نقابه",
    "انهاء العقد",
    "محدد المده",
)

_COMPENSATION_CONTEXT_MARKERS = (
    "اصابه",
    "عجز",
    "وفاه",
    "فصل",
    "تعسفي",
    "عقد",
    "اجر",
    "راتب",
    "ضرر",
    "عمل",
)

_DEDUCTION_CONTEXT_MARKERS = (
    "سلفه",
    "تلف",
    "اتلف",
    "خطا",
    "حكم",
    "محكمه",
    "دين",
    "نفقه",
    "غرامه",
    "نسبه",
    "قيمه",
    "سبب",
)

_CONTRACT_DETAIL_MARKERS = (
    "شفهي",
    "مكتوب",
    "محدد المده",
    "غير محدد",
    "تجربه",
    "انهاء",
    "اشعار",
    "اجر",
    "راتب",
    "مكان",
    "نقل",
    "تدريب",
    "جماعي",
)


def _contains_any(
    normalized_text: str,
    phrases: Iterable[str],
) -> bool:
    return any(
        normalize_arabic(phrase) in normalized_text
        for phrase in phrases
    )


def _route_question(
    normalized_question: str,
) -> tuple[str, str]:
    """
    Decide whether retrieval is appropriate before searching the KG.

    The router is deliberately conservative. Strong non-labour-domain
    phrases cause abstention. Broad labour questions that omit a legally
    decisive fact cause clarification. Everything else proceeds to
    retrieval.
    """

    for group in _OUT_OF_SCOPE_GROUPS:
        if _contains_any(normalized_question, group):
            return (
                "abstain",
                "The question belongs to a legal domain outside the "
                "Jordanian Labor Law knowledge graph.",
            )

    if (
        "اجازه" in normalized_question
        and not _contains_any(
            normalized_question,
            _LEAVE_SPECIFIERS,
        )
    ):
        return (
            "clarify",
            "The type of leave is not specified.",
        )

    if (
        _contains_any(
            normalized_question,
            ("يفصلني", "فصلي", "فصل العامل"),
        )
        and not _contains_any(
            normalized_question,
            _DISMISSAL_DETAIL_MARKERS,
        )
    ):
        return (
            "clarify",
            "The reason and circumstances of dismissal are not specified.",
        )

    if (
        "تعويض" in normalized_question
        and not _contains_any(
            normalized_question,
            _COMPENSATION_CONTEXT_MARKERS,
        )
    ):
        return (
            "clarify",
            "The legal event giving rise to compensation is not specified.",
        )

    if (
        _contains_any(
            normalized_question,
            ("حسم", "خصم", "اقتطاع"),
        )
        and not _contains_any(
            normalized_question,
            _DEDUCTION_CONTEXT_MARKERS,
        )
    ):
        return (
            "clarify",
            "The reason and amount of the wage deduction are not specified.",
        )

    if (
        "عقد عمل" in normalized_question
        and _contains_any(
            normalized_question,
            ("شو حقوقي", "ما حقوقي", "حقوقي"),
        )
        and not _contains_any(
            normalized_question,
            _CONTRACT_DETAIL_MARKERS,
        )
    ):
        return (
            "clarify",
            "The contract issue or requested right is not specified.",
        )

    return ("retrieve", "")


def analyze_legal_question(
    question: str,
) -> LegalQuestionAnalysis:
    normalized_question = normalize_arabic(
        question
    )

    behavior, behavior_reason = _route_question(
        normalized_question
    )

    matched_profiles = [
        profile
        for profile in ISSUE_PROFILES
        if _profile_matches(
            normalized_question,
            profile,
        )
    ]

    tokens = {
        token
        for token in normalized_question.split()
        if (
            len(token) > 1
            and token not in _STOPWORDS
        )
    }

    numeric_tokens = {
        token
        for token in normalized_question.split()
        if token.isdigit()
    }

    return LegalQuestionAnalysis(
        original_question=question,
        normalized_question=normalized_question,
        issue_ids=tuple(
            profile.issue_id
            for profile in matched_profiles
        ),
        preferred_concepts=_deduplicate(
            concept
            for profile in matched_profiles
            for concept in profile.preferred_concepts
        ),
        query_expansion_terms=_deduplicate(
            term
            for profile in matched_profiles
            for term in profile.query_expansion_terms
        ),
        article_anchor_phrases=_deduplicate(
            phrase
            for profile in matched_profiles
            for phrase in profile.article_anchor_phrases
        ),
        primary_article_anchor_phrases=_deduplicate(
            phrase
            for profile in matched_profiles
            for phrase in profile.primary_article_anchor_phrases
        ),
        meaningful_tokens=frozenset(tokens),
        numeric_tokens=frozenset(
            numeric_tokens
        ),
        max_final_articles=(
            max(
                (
                    profile.max_final_articles
                    for profile in matched_profiles
                ),
                default=1,
            )
        ),
        behavior=behavior,
        behavior_reason=behavior_reason,
    )


def article_question_relevance(
    analysis: LegalQuestionAnalysis,
    article_text: str,
) -> float:
    """
    Deterministic Arabic legal-question/article relevance in [0, 1].

    This score does not use article numbers. It combines question-token
    coverage, meaningful phrase overlap, issue-specific legal anchors, and
    numeric overlap. It is used only as one feature in hybrid ranking.
    """

    normalized_article = normalize_arabic(
        article_text
    )

    if not normalized_article:
        return 0.0

    article_tokens = set(
        normalized_article.split()
    )

    if analysis.meaningful_tokens:
        token_coverage = (
            len(
                analysis.meaningful_tokens
                & article_tokens
            )
            / len(analysis.meaningful_tokens)
        )
    else:
        token_coverage = 0.0

    question_tokens = [
        token
        for token in (
            analysis
            .normalized_question
            .split()
        )
        if (
            len(token) > 1
            and token not in _STOPWORDS
        )
    ]

    question_phrases: list[str] = []
    for size in (4, 3, 2):
        for index in range(
            0,
            max(0, len(question_tokens) - size + 1),
        ):
            phrase = " ".join(
                question_tokens[
                    index:index + size
                ]
            )
            question_phrases.append(
                phrase
            )

    meaningful_phrases = [
        phrase
        for phrase in question_phrases
        if len(phrase) >= 7
    ]

    if meaningful_phrases:
        phrase_matches = sum(
            phrase in normalized_article
            for phrase in meaningful_phrases
        )
        phrase_score = min(
            1.0,
            phrase_matches / 3.0,
        )
    else:
        phrase_score = 0.0

    normalized_anchors = [
        normalize_arabic(phrase)
        for phrase in (
            analysis.article_anchor_phrases
        )
    ]

    if normalized_anchors:
        anchor_matches = sum(
            anchor in normalized_article
            for anchor in normalized_anchors
        )
        anchor_score = (
            anchor_matches
            / len(normalized_anchors)
        )
    else:
        anchor_score = 0.0

    article_numbers = {
        token
        for token in normalized_article.split()
        if token.isdigit()
    }

    if analysis.numeric_tokens:
        number_score = (
            len(
                analysis.numeric_tokens
                & article_numbers
            )
            / len(analysis.numeric_tokens)
        )
    else:
        number_score = 0.0

    score = (
        0.30 * token_coverage
        + 0.20 * phrase_score
        + 0.45 * anchor_score
        + 0.05 * number_score
    )

    normalized_primary_anchors = [
        normalize_arabic(phrase)
        for phrase in analysis.primary_article_anchor_phrases
    ]

    if normalized_primary_anchors:
        primary_anchor_score = (
            sum(
                anchor in normalized_article
                for anchor in normalized_primary_anchors
            )
            / len(normalized_primary_anchors)
        )

        # Preserve the established relevance score while giving the
        # principal provision a modest ordering advantage over a related
        # consequence provision.  This is article-number agnostic.
        score = (
            0.82 * score
            + 0.18 * primary_anchor_score
        )

    return max(
        0.0,
        min(1.0, score),
    )
