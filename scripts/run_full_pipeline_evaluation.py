from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from openai import OpenAI

from app.config import get_settings

DEFAULT_BENCHMARK = Path('/app/data/benchmarks/jordan_labor_law_fresh_final_40_v2_3.json')
DEFAULT_RESULTS_DIR = Path('/app/data/model_evaluations')
ARABIC_RE = re.compile(r'[\u0600-\u06FF]')
LATIN_RE = re.compile(r'[A-Za-z]')
CITATION_RE = re.compile(r'\[\s*المادة\s+([0-9٠-٩۰-۹]+)\s*\]')
DIGIT_TRANS = str.maketrans('٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹', '01234567890123456789')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Run the complete Jordanian Labor Law model-comparison pipeline. '
            'By default this performs three independent 40-question runs and '
            'writes per-run JSON plus one mean/std aggregate JSON.'
        )
    )
    parser.add_argument(
        '--model-name',
        default=None,
        help=(
            'Optional display label. Defaults to PIPELINE_LLM_MODEL from the '
            'active API/container configuration.'
        ),
    )
    parser.add_argument('--benchmark', type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument(
        '--runs',
        type=int,
        default=3,
        help='Independent repetitions for the final comparison (default: 3).',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Directory for run-1.json ... final-summary.json.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help=(
            'Optional single-run output path. Supplying this forces '
            'one run and is useful for smoke/diagnostic tests.'
        ),
    )
    parser.add_argument('--base-url', default='http://localhost:8000')
    parser.add_argument(
        '--judge-model',
        default=os.getenv('EVALUATION_JUDGE_MODEL', 'gpt-5.4-mini'),
    )
    parser.add_argument('--timeout', type=float, default=240.0)
    parser.add_argument('--delay', type=float, default=0.0)
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--stop-on-error', action='store_true')
    return parser.parse_args()


class ExecutionIntegrityError(RuntimeError):
    """Raised when a model-comparison run stops using the configured LLM."""


def validate_execution_integrity(
    *,
    case_id: str,
    expected_behavior: str,
    retrieval: dict[str, Any],
    generation: dict[str, Any],
    expected_model: str,
) -> None:
    """Fail closed when a benchmark case silently leaves the LLM path.

    The planner flag is observable in retrieval.v2. Reranker/provider failures are never replaced by deterministic fallbacks.
    Provider failures remain execution errors, while genuine
    model-quality failures (for example invalid citations after a successful
    model call) remain scoreable outcomes rather than infrastructure errors.
    """
    decision = retrieval.get('decision')
    if not isinstance(decision, dict):
        raise ExecutionIntegrityError(
            f'{case_id}: retrieval output has no decision object.'
        )

    if decision.get('planner_used') is not True:
        raise ExecutionIntegrityError(
            f'{case_id}: query planner was not used; deterministic fallback '
            'would make the model comparison invalid.'
        )

    warnings = generation.get('warnings')
    warning_values = (
        [str(value) for value in warnings]
        if isinstance(warnings, list)
        else []
    )
    infrastructure_markers = (
        'Pipeline answer generation failed.',
        'PIPELINE_ANSWER_ENABLED is false',
        'no pipeline LLM provider is available',
    )
    if any(
        marker in warning
        for warning in warning_values
        for marker in infrastructure_markers
    ):
        raise ExecutionIntegrityError(
            f'{case_id}: answer-generation provider failure detected: '
            + ' | '.join(warning_values)
        )

    if expected_behavior == 'retrieve' and generation.get('status') == 'generated':
        actual_model = str(generation.get('model') or '')
        if actual_model != expected_model:
            raise ExecutionIntegrityError(
                f'{case_id}: generated answer reports model={actual_model!r}; '
                f'expected {expected_model!r}.'
            )


def load_json(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8-sig') as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object: {path}')
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_slug(value: str) -> str:
    slug = re.sub(r'[^A-Za-z0-9._-]+', '_', value.strip()).strip('._-')
    if not slug:
        raise ValueError('model-name must contain at least one safe character.')
    return slug.lower()


def ordered_unique(values: list[int]) -> list[int]:
    return list(dict.fromkeys(values))


def to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).translate(DIGIT_TRANS))
    except (TypeError, ValueError):
        return None


def get_behavior(retrieval: dict[str, Any]) -> str:
    decision = retrieval.get('decision')
    if isinstance(decision, dict):
        behavior = decision.get('behavior')
        if behavior:
            return str(behavior)
    behavior = retrieval.get('behavior')
    return str(behavior or '')


def get_articles(retrieval: dict[str, Any]) -> list[dict[str, Any]]:
    articles = retrieval.get('articles')
    if isinstance(articles, list):
        return [item for item in articles if isinstance(item, dict)]
    nested = retrieval.get('retrieval')
    if isinstance(nested, dict):
        hits = nested.get('article_hits')
        if isinstance(hits, list):
            return [item for item in hits if isinstance(item, dict)]
    return []


def actual_article_numbers(retrieval: dict[str, Any]) -> list[int]:
    numbers: list[int] = []
    diagnostics = retrieval.get('diagnostics')
    if isinstance(diagnostics, dict):
        for key in ('article_numbers', 'top_article_numbers'):
            values = diagnostics.get(key)
            if isinstance(values, list):
                numbers.extend(v for x in values if (v := to_int(x)) is not None)
                if numbers:
                    return ordered_unique(numbers)
    for article in get_articles(retrieval):
        number = to_int(article.get('article_number'))
        if number is not None:
            numbers.append(number)
    return ordered_unique(numbers)


def extract_citations(generation: dict[str, Any]) -> list[int]:
    values = generation.get('cited_article_numbers')
    numbers: list[int] = []
    if isinstance(values, list):
        numbers.extend(v for x in values if (v := to_int(x)) is not None)
    answer = str(generation.get('answer_ar') or '')
    numbers.extend(int(match.translate(DIGIT_TRANS)) for match in CITATION_RE.findall(answer))
    return ordered_unique(numbers)


def is_arabic_only(text: str) -> bool:
    clean = text.strip()
    return bool(clean and ARABIC_RE.search(clean) and not LATIN_RE.search(clean))


class ModelContractFailure(RuntimeError):
    """
    A model-produced response violated the frozen structured-output contract.

    This is scored as a model/pipeline failure rather than an infrastructure
    execution failure. Provider/network failures remain ordinary exceptions.
    """

    def __init__(
        self,
        *,
        url: str,
        detail: str,
        elapsed: float,
    ) -> None:
        super().__init__(detail)
        self.url = url
        self.detail = detail
        self.elapsed = elapsed

        endpoint = url.split('?', 1)[0].rstrip('/')

        if endpoint.endswith('/retrieve'):
            self.stage = 'retrieval'
        elif endpoint.endswith('/generate'):
            self.stage = 'generation'
        else:
            self.stage = 'pipeline'


def _http_error_detail(response: requests.Response) -> str:
    try:
        value = response.json()
    except Exception:
        return response.text[:4000]

    if isinstance(value, dict):
        detail = value.get('detail')
        if detail is not None:
            return str(detail)

    return response.text[:4000]


def _is_model_contract_failure(
    status_code: int,
    detail: str,
) -> bool:
    """
    Distinguish model-produced contract/validation failures from
    infrastructure/provider execution failures.

    Model contract failures are scored as failed benchmark cases.
    Infrastructure failures retain the checkpoint/retry policy.
    """
    if status_code != 500:
        return False

    normalized = detail.lower()

    infrastructure_markers = (
        'cohere request failed.',
        'status=429',
        'status=500',
        'status=502',
        'status=503',
        'status=504',
        'timeout',
        'timed out',
        'connection error',
        'connectionerror',
    )

    if any(
        marker in normalized
        for marker in infrastructure_markers
    ):
        return False

    # Pydantic / structured JSON contract violation.
    if 'structured-output validation failed' in normalized:
        return True

    # The legal query planner performs additional semantic contract
    # validation after the structured JSON has passed Pydantic.
    # Examples include an abstain plan containing atomic issues or
    # requesting articles. Those ValueErrors are model-output failures,
    # not provider/infrastructure failures.
    model_stage_markers = (
        'query planner failed.',
        'article reranker failed.',
        'answer generation failed.',
        'citation-repair retry failed.',
    )

    validation_error_markers = (
        'error=valueerror:',
        'error=validationerror:',
        'validationerror:',
    )

    return (
        any(
            marker in normalized
            for marker in model_stage_markers
        )
        and any(
            marker in normalized
            for marker in validation_error_markers
        )
    )


def post_json(session: requests.Session, url: str, payload: dict[str, Any], timeout: float) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    response = session.post(url, json=payload, timeout=timeout)
    elapsed = time.perf_counter() - started

    if response.status_code >= 400:
        detail = _http_error_detail(response)

        if _is_model_contract_failure(
            response.status_code,
            detail,
        ):
            raise ModelContractFailure(
                url=url,
                detail=detail,
                elapsed=elapsed,
            )

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise requests.HTTPError(
                f"{exc} | response_detail={detail[:2000]}"
            ) from exc

    value = response.json()
    if not isinstance(value, dict):
        raise ValueError(f'Expected JSON object from {url}')
    return value, elapsed


def evaluate_retrieval(
    case: dict[str, Any],
    retrieval: dict[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    expected_behavior = str(case.get('expected_behavior', 'retrieve'))
    actual_behavior = get_behavior(retrieval)
    actual = actual_article_numbers(retrieval)[:5]
    required = [int(x) for x in case.get('required_articles', [])]
    acceptable = [int(x) for x in case.get('acceptable_articles', required)]

    routing_correct = actual_behavior == expected_behavior

    if expected_behavior == 'retrieve':
        required_set = set(required)
        relevant_set = set(required) | set(acceptable)
        actual_set = set(actual)
        hit_at_1 = bool(actual and actual[0] in relevant_set)
        article_recall_at_5 = (
            len(actual_set & required_set) / len(required_set)
            if required_set else 1.0
        )
        article_precision = (
            len(actual_set & relevant_set) / len(actual_set)
            if actual_set else 0.0
        )
    else:
        hit_at_1 = False
        article_recall_at_5 = 0.0
        article_precision = 0.0

    return {
        'expected_behavior': expected_behavior,
        'actual_behavior': actual_behavior,
        'required_articles': required,
        'acceptable_articles': acceptable,
        'actual_articles_at_5': actual,
        'routing_correct': routing_correct,
        'hit_at_1': hit_at_1,
        'article_recall_at_5': round(article_recall_at_5, 6),
        'article_precision': round(article_precision, 6),
        'elapsed_seconds': round(elapsed, 3),
    }

def model_contract_failure_evaluations(
    case: dict[str, Any],
    *,
    stage: str,
    retrieval: dict[str, Any] | None,
    retrieval_elapsed: float | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Convert a deterministic model contract violation into a completed
    failed benchmark case.

    Retrieval-stage contract failures:
      - penalize routing/retrieval
      - do not invent generation-quality scores because generation
        was never reached

    Generation-stage contract failures:
      - preserve the real retrieval metrics
      - score generation as failed
    """
    expected_behavior = str(
        case.get('expected_behavior', 'retrieve')
    )

    required = [
        int(x)
        for x in case.get('required_articles', [])
    ]

    acceptable = [
        int(x)
        for x in case.get(
            'acceptable_articles',
            required,
        )
    ]

    if (
        stage == 'generation'
        and retrieval is not None
        and retrieval_elapsed is not None
    ):
        retrieval_eval = evaluate_retrieval(
            case,
            retrieval,
            retrieval_elapsed,
        )
    else:
        retrieval_eval = {
            'expected_behavior': expected_behavior,
            'actual_behavior': 'model_contract_failure',
            'required_articles': required,
            'acceptable_articles': acceptable,
            'actual_articles_at_5': [],
            'routing_correct': False,
            'hit_at_1': False,
            'article_recall_at_5': 0.0,
            'article_precision': 0.0,
            'elapsed_seconds': None,
        }

    generation_reached = stage == 'generation'

    generation_eval = {
        'status': 'model_contract_failure',
        'answer_ar': '',
        'cited_article_numbers': [],
        'out_of_scope_response_correct': (
            False
            if (
                generation_reached
                and expected_behavior == 'abstain'
            )
            else None
        ),
        'correctness_grade': (
            0 if generation_reached else None
        ),
        'faithfulness_grade': (
            0 if generation_reached else None
        ),
        'correctness': (
            0.0 if generation_reached else None
        ),
        'faithfulness': (
            0.0 if generation_reached else None
        ),
        'citation_validity': (
            0.0 if generation_reached else None
        ),
        'citation_recall': (
            0.0 if generation_reached else None
        ),
        'elapsed_seconds': None,
        'judge': None,
    }

    return retrieval_eval, generation_eval


def response_text(response: Any) -> str:
    text = str(getattr(response, 'output_text', '') or '')
    if text:
        return text
    raise RuntimeError('Judge returned no output text.')


def judge_generation(
    client: OpenAI,
    judge_model: str,
    case: dict[str, Any],
    retrieval: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    evidence = [
        {
            'article_number': to_int(article.get('article_number')),
            'text': str(article.get('text') or ''),
        }
        for article in get_articles(retrieval)
    ]

    schema = {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'correctness_grade': {'type': 'integer', 'enum': [0, 1, 2]},
            'faithfulness_grade': {'type': 'integer', 'enum': [0, 1, 2]},
            'missing_required_facts': {
                'type': 'array',
                'items': {'type': 'string'},
            },
            'unsupported_claims': {
                'type': 'array',
                'items': {'type': 'string'},
            },
            'reason_ar': {'type': 'string'},
        },
        'required': [
            'correctness_grade',
            'faithfulness_grade',
            'missing_required_facts',
            'unsupported_claims',
            'reason_ar',
        ],
    }

    payload = {
        'question': case.get('question'),
        'required_facts': [str(x) for x in case.get('required_facts', [])],
        'retrieved_legal_evidence': evidence,
        'generated_answer': str(generation.get('answer_ar') or ''),
        'generated_citations': extract_citations(generation),
    }

    instructions = """
You evaluate an Arabic legal question-answering system.

Use only:
1. the user question;
2. the required facts;
3. the retrieved legal evidence;
4. the generated answer.

Correctness:
2 = all required facts are correct and the answer fully answers the question.
1 = the answer is partially correct but misses or weakens an important part.
0 = the answer is substantially incorrect, misleading, or does not answer.

Faithfulness:
2 = every substantive legal claim is supported by the retrieved evidence.
1 = the main answer is supported, but there is a minor unsupported addition,
    overstatement, or imprecision.
0 = there is a major unsupported or contradictory legal claim.

Do not use external legal knowledge.
Do not penalize wording, style, or harmless paraphrasing differences.
Return the required strict JSON only.
""".strip()

    response = client.responses.create(
        model=judge_model,
        instructions=instructions,
        input=json.dumps(payload, ensure_ascii=False, separators=(',', ':')),
        text={
            'format': {
                'type': 'json_schema',
                'name': 'simple_legal_generation_evaluation',
                'schema': schema,
                'strict': True,
            }
        },
        store=False,
    )

    result = json.loads(response_text(response))
    return {
        'correctness_grade': int(result['correctness_grade']),
        'faithfulness_grade': int(result['faithfulness_grade']),
        'missing_required_facts': [
            str(x) for x in result.get('missing_required_facts', [])
        ],
        'unsupported_claims': [
            str(x) for x in result.get('unsupported_claims', [])
        ],
        'reason_ar': str(result.get('reason_ar', '')),
    }


def evaluate_generation(
    case: dict[str, Any],
    retrieval: dict[str, Any],
    generation: dict[str, Any],
    elapsed: float,
    client: OpenAI,
    judge_model: str,
) -> dict[str, Any]:
    expected_behavior = str(case.get('expected_behavior', 'retrieve'))
    answer = str(generation.get('answer_ar') or '').strip()
    status = str(generation.get('status') or '')
    citations = extract_citations(generation)
    retrieved_set = set(actual_article_numbers(retrieval))
    required_set = set(
        int(x)
        for x in case.get(
            'required_citations',
            case.get('required_articles', []),
        )
    )
    citation_set = set(citations)

    citation_validity = (
        len(citation_set & retrieved_set) / len(citation_set)
        if citation_set else (1.0 if expected_behavior == 'abstain' else 0.0)
    )
    citation_recall = (
        len(citation_set & required_set) / len(required_set)
        if required_set else 1.0
    )

    if expected_behavior == 'abstain':
        response_correct = (
            get_behavior(retrieval) == 'abstain'
            and status == 'out_of_scope'
            and bool(answer)
            and is_arabic_only(answer)
            and not citations
        )
        return {
            'status': status,
            'answer_ar': answer,
            'cited_article_numbers': citations,
            'out_of_scope_response_correct': response_correct,
            'correctness_grade': None,
            'faithfulness_grade': None,
            'correctness': None,
            'faithfulness': None,
            'citation_validity': round(citation_validity, 6),
            'citation_recall': round(citation_recall, 6),
            'elapsed_seconds': round(elapsed, 3),
            'judge': None,
        }

    judge = judge_generation(
        client,
        judge_model,
        case,
        retrieval,
        generation,
    )
    correctness_grade = judge['correctness_grade']
    faithfulness_grade = judge['faithfulness_grade']

    return {
        'status': status,
        'answer_ar': answer,
        'cited_article_numbers': citations,
        'out_of_scope_response_correct': None,
        'correctness_grade': correctness_grade,
        'faithfulness_grade': faithfulness_grade,
        'correctness': round(correctness_grade / 2.0, 6),
        'faithfulness': round(faithfulness_grade / 2.0, 6),
        'citation_validity': round(citation_validity, 6),
        'citation_recall': round(citation_recall, 6),
        'elapsed_seconds': round(elapsed, 3),
        'judge': judge,
    }

def mean(rows: list[dict[str, Any]], path: tuple[str, ...], default: float = 0.0) -> float:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if isinstance(value, bool):
            values.append(float(value))
        elif isinstance(value, (int, float)):
            values.append(float(value))
    return round(statistics.fmean(values), 6) if values else default


def make_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if not row.get('error')]
    retrieve = [row for row in completed if row['expected_behavior'] == 'retrieve']
    abstain = [row for row in completed if row['expected_behavior'] == 'abstain']
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in completed:
        by_type[str(row.get('test_type', 'unknown'))].append(row)

    return {
        'questions_requested': len(rows),
        'questions_completed': len(completed),
        'questions_failed_to_run': len(rows) - len(completed),
        'end_to_end_success_rate': mean(completed, ('end_to_end_pass',)),
        'retrieval': {
            'routing_accuracy': mean(
                completed,
                ('retrieval_evaluation', 'routing_correct'),
            ),
            'hit_at_1': mean(
                retrieve,
                ('retrieval_evaluation', 'hit_at_1'),
            ),
            'article_recall_at_5': mean(
                retrieve,
                ('retrieval_evaluation', 'article_recall_at_5'),
            ),
            'article_precision': mean(
                retrieve,
                ('retrieval_evaluation', 'article_precision'),
            ),
            'out_of_scope_accuracy': mean(
                abstain,
                ('retrieval_evaluation', 'routing_correct'),
            ),
            'mean_latency_seconds': mean(
                completed,
                ('retrieval_evaluation', 'elapsed_seconds'),
            ),
        },
        'generation': {
            'correctness': mean(
                retrieve,
                ('generation_evaluation', 'correctness'),
            ),
            'faithfulness': mean(
                retrieve,
                ('generation_evaluation', 'faithfulness'),
            ),
            'citation_validity': mean(
                retrieve,
                ('generation_evaluation', 'citation_validity'),
            ),
            'citation_recall': mean(
                retrieve,
                ('generation_evaluation', 'citation_recall'),
            ),
            'out_of_scope_response_accuracy': mean(
                abstain,
                ('generation_evaluation', 'out_of_scope_response_correct'),
            ),
            'mean_latency_seconds': mean(
                completed,
                ('generation_evaluation', 'elapsed_seconds'),
            ),
        },
        'by_test_type': {
            test_type: {
                'count': len(group),
                'end_to_end_success_rate': mean(
                    group,
                    ('end_to_end_pass',),
                ),
                'routing_accuracy': mean(
                    group,
                    ('retrieval_evaluation', 'routing_correct'),
                ),
            }
            for test_type, group in sorted(by_type.items())
        },
    }

def build_output(
    args: argparse.Namespace,
    benchmark: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    run_index: int,
    runs_requested: int,
    model_name: str,
) -> dict[str, Any]:
    settings = get_settings()

    return {
        'schema_version': 'full-pipeline-evaluation.v4',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'model_name': model_name,
        'provider': settings.pipeline_llm_provider,
        'run_index': run_index,
        'runs_requested': runs_requested,
        'judge_model': args.judge_model,
        'execution_integrity': {
            'strict_llm_execution': True,
            'checkpoint_resume_enabled': True,
        },
        'pipeline_models': {
            'query_planner': {
                'provider': settings.pipeline_llm_provider,
                'model': settings.pipeline_llm_model,
            },
            'route_verifier': {
                'provider': settings.pipeline_llm_provider,
                'model': settings.pipeline_llm_model,
            },
            'article_reranker': {
                'provider': settings.pipeline_llm_provider,
                'model': settings.pipeline_llm_model,
            },
            'answer_generator': {
                'provider': settings.pipeline_llm_provider,
                'model': settings.pipeline_llm_model,
            },
            'citation_retry': {
                'provider': settings.pipeline_llm_provider,
                'model': settings.pipeline_llm_model,
            },
            'embedding': {
                'provider': 'openai',
                'model': settings.openai_embedding_model,
            },
            'judge': {
                'provider': 'openai',
                'model': args.judge_model,
            },
        },
        'frozen_comparison_settings': {
            'planner_max_output_tokens': settings.planner_max_output_tokens,
            'reranker_max_output_tokens': settings.reranker_max_output_tokens,
            'generator_max_output_tokens': settings.generator_max_output_tokens,
            'reranker_candidate_limit': getattr(
                settings, 'reranker_candidate_limit', 12
            ),
            'reranker_total_char_budget': getattr(
                settings, 'reranker_total_char_budget', 12000
            ),
            'reranker_article_char_limit': int(
                os.getenv('PIPELINE_RERANK_ARTICLE_CHAR_LIMIT', '2500')
            ),
            'ollama_num_ctx': getattr(settings, 'ollama_num_ctx', None),
            'embedding_model': settings.openai_embedding_model,
            'judge_model': args.judge_model,
        },
        'benchmark': {
            'name': benchmark.get('benchmark_name'),
            'version': benchmark.get('benchmark_version'),
            'sha256': sha256_file(args.benchmark),
            'path': str(args.benchmark),
            'frozen': benchmark.get('frozen'),
        },
        'summary': make_summary(rows),
        'results': rows,
    }


def save_output(path: Path, output: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write('\n')
    temp.replace(path)



def _nested_number(value: dict[str, Any], path: tuple[str, ...]) -> float | None:
    current: Any = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if isinstance(current, bool):
        return float(current)
    if isinstance(current, (int, float)):
        return float(current)
    return None


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
    return {
        'mean': round(statistics.fmean(values), 6),
        # Sample standard deviation across independent benchmark runs.
        'std': round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
        'min': round(min(values), 6),
        'max': round(max(values), 6),
    }


def _aggregate_path(
    outputs: list[dict[str, Any]],
    path: tuple[str, ...],
) -> dict[str, float]:
    values: list[float] = []
    for output in outputs:
        value = _nested_number(output.get('summary', {}), path)
        if value is not None:
            values.append(value)
    return _stats(values)


def build_aggregate_output(
    *,
    args: argparse.Namespace,
    benchmark: dict[str, Any],
    run_outputs: list[dict[str, Any]],
    run_paths: list[Path],
    runs_requested: int,
    selected_count: int,
    model_name: str,
) -> dict[str, Any]:
    settings = get_settings()

    valid_runs = [
        output
        for output in run_outputs
        if output.get('summary', {}).get('questions_completed') == selected_count
        and output.get('summary', {}).get('questions_failed_to_run') == 0
    ]
    valid = len(valid_runs) == runs_requested

    retrieval_metrics = (
        'routing_accuracy',
        'hit_at_1',
        'article_recall_at_5',
        'article_precision',
        'out_of_scope_accuracy',
        'mean_latency_seconds',
    )
    generation_metrics = (
        'correctness',
        'faithfulness',
        'citation_validity',
        'citation_recall',
        'out_of_scope_response_accuracy',
        'mean_latency_seconds',
    )

    all_test_types: set[str] = set()
    for output in valid_runs:
        by_type = output.get('summary', {}).get('by_test_type', {})
        if isinstance(by_type, dict):
            all_test_types.update(str(key) for key in by_type)

    aggregate = {
        'end_to_end_success_rate': _aggregate_path(
            valid_runs, ('end_to_end_success_rate',)
        ),
        'retrieval': {
            metric: _aggregate_path(valid_runs, ('retrieval', metric))
            for metric in retrieval_metrics
        },
        'generation': {
            metric: _aggregate_path(valid_runs, ('generation', metric))
            for metric in generation_metrics
        },
        'by_test_type': {
            test_type: {
                'end_to_end_success_rate': _aggregate_path(
                    valid_runs,
                    ('by_test_type', test_type, 'end_to_end_success_rate'),
                ),
                'routing_accuracy': _aggregate_path(
                    valid_runs,
                    ('by_test_type', test_type, 'routing_accuracy'),
                ),
            }
            for test_type in sorted(all_test_types)
        },
    }

    return {
        'schema_version': 'model-comparison-aggregate.v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'valid' if valid else 'incomplete',
        'provider': settings.pipeline_llm_provider,
        'model': settings.pipeline_llm_model,
        'model_name': model_name,
        'runs_requested': runs_requested,
        'runs_valid': len(valid_runs),
        'questions_per_run': selected_count,
        'std_definition': 'sample standard deviation across independent runs',
        'benchmark': {
            'name': benchmark.get('benchmark_name'),
            'version': benchmark.get('benchmark_version'),
            'sha256': sha256_file(args.benchmark),
            'path': str(args.benchmark),
            'frozen': benchmark.get('frozen'),
        },
        'pipeline_models': (run_outputs[0].get('pipeline_models') if run_outputs else {}),
        'frozen_comparison_settings': (
            run_outputs[0].get('frozen_comparison_settings')
            if run_outputs
            else {}
        ),
        'individual_runs': [
            {
                'run_index': output.get('run_index'),
                'path': str(path),
                'summary': output.get('summary'),
            }
            for output, path in zip(run_outputs, run_paths)
        ],
        'aggregate': aggregate if valid else None,
        'invalid_reason': (
            None
            if valid
            else (
                f'Expected {runs_requested} complete runs of {selected_count} questions; '
                f'only {len(valid_runs)} runs were valid. No partial mean is reported.'
            )
        ),
    }

def print_row(
    row: dict[str, Any],
    index: int,
    total: int,
) -> None:
    if row.get('error'):
        print(
            f'[{index:02d}/{total:02d}] '
            f'{row["id"]} ERROR | {row["error"]}'
        )
        return

    if row.get('model_contract_failure'):
        print(
            f'[{index:02d}/{total:02d}] '
            f'{row["id"]} FAIL '
            f'| MODEL_CONTRACT_FAILURE '
            f'| Stage={row.get("model_contract_stage", "unknown")}'
        )
        return

    r = row['retrieval_evaluation']
    g = row['generation_evaluation']
    label = 'PASS' if row['end_to_end_pass'] else 'FAIL'

    if row['expected_behavior'] == 'abstain':
        print(
            f'[{index:02d}/{total:02d}] {row["id"]} {label} '
            f'| Route={int(bool(r["routing_correct"]))} '
            f'| OOS={int(bool(g["out_of_scope_response_correct"]))}'
        )
        return

    print(
        f'[{index:02d}/{total:02d}] {row["id"]} {label} '
        f'| Route={int(bool(r["routing_correct"]))} '
        f'| Hit@1={int(bool(r["hit_at_1"]))} '
        f'| Recall@5={r["article_recall_at_5"]:.2f} '
        f'| Precision={r["article_precision"]:.2f} '
        f'| Correct={g["correctness_grade"]}/2 '
        f'| Faith={g["faithfulness_grade"]}/2 '
        f'| CitValid={g["citation_validity"]:.2f} '
        f'| CitRecall={g["citation_recall"]:.2f}'
    )


def _case_identity(case: dict[str, Any], index: int) -> dict[str, str]:
    return {
        'id': str(case.get('id', f'question_{index}')),
        'question': str(case.get('question', '')),
        'test_type': str(case.get('test_type', 'unknown')),
        'category': str(case.get('category', 'unknown')),
        'difficulty': str(case.get('difficulty', 'unknown')),
        'expected_behavior': str(case.get('expected_behavior', 'retrieve')),
    }


def _load_resume_checkpoint(
    *,
    args: argparse.Namespace,
    benchmark: dict[str, Any],
    selected: list[dict[str, Any]],
    output_path: Path,
    run_index: int,
    runs_requested: int,
    model_name: str,
) -> tuple[list[dict[str, Any]], bool]:
    """Load a compatible run checkpoint and return (rows, is_complete).

    Resume is fail-closed: an existing file must match the active benchmark,
    provider/model, judge, run index, requested repetitions, and the exact
    selected question prefix. A trailing infrastructure-error row is discarded
    so that the failed question is retried; completed rows are never rerun.
    """
    if not output_path.exists():
        return [], False

    checkpoint = load_json(output_path)
    settings = get_settings()

    expected_benchmark_sha = sha256_file(args.benchmark)
    observed_benchmark = checkpoint.get('benchmark')
    if not isinstance(observed_benchmark, dict):
        raise RuntimeError(
            f'Refusing to resume {output_path}: missing benchmark metadata.'
        )
    if observed_benchmark.get('sha256') != expected_benchmark_sha:
        raise RuntimeError(
            f'Refusing to resume {output_path}: benchmark SHA-256 mismatch.'
        )

    checks = {
        'run_index': (checkpoint.get('run_index'), run_index),
        'runs_requested': (checkpoint.get('runs_requested'), runs_requested),
        'model_name': (str(checkpoint.get('model_name') or ''), model_name),
        'provider': (
            str(checkpoint.get('provider') or ''),
            settings.pipeline_llm_provider,
        ),
        'judge_model': (
            str(checkpoint.get('judge_model') or ''),
            args.judge_model,
        ),
    }
    for label, (observed, expected) in checks.items():
        if observed != expected:
            raise RuntimeError(
                f'Refusing to resume {output_path}: {label} mismatch '
                f'(checkpoint={observed!r}, active={expected!r}).'
            )

    pipeline_models = checkpoint.get('pipeline_models')
    if not isinstance(pipeline_models, dict):
        raise RuntimeError(
            f'Refusing to resume {output_path}: missing pipeline model metadata.'
        )
    for stage in (
        'query_planner',
        'route_verifier',
        'article_reranker',
        'answer_generator',
        'citation_retry',
    ):
        stage_meta = pipeline_models.get(stage)
        if not isinstance(stage_meta, dict):
            raise RuntimeError(
                f'Refusing to resume {output_path}: missing {stage} metadata.'
            )
        observed_provider = str(stage_meta.get('provider') or '')
        observed_model = str(stage_meta.get('model') or '')
        if (
            observed_provider != settings.pipeline_llm_provider
            or observed_model != settings.pipeline_llm_model
        ):
            raise RuntimeError(
                f'Refusing to resume {output_path}: active pipeline model '
                f'does not match checkpoint at stage {stage}.'
            )

    rows = checkpoint.get('results')
    if not isinstance(rows, list):
        raise RuntimeError(
            f'Refusing to resume {output_path}: results is not a list.'
        )
    rows = [dict(row) for row in rows if isinstance(row, dict)]
    if len(rows) > len(selected):
        raise RuntimeError(
            f'Refusing to resume {output_path}: checkpoint contains more '
            'questions than the active selection.'
        )

    # The only legitimate failed row in strict mode is the final attempted
    # question. Remove it so the exact failed question is retried on resume.
    error_indexes = [i for i, row in enumerate(rows) if row.get('error')]
    if error_indexes:
        if error_indexes != [len(rows) - 1]:
            raise RuntimeError(
                f'Refusing to resume {output_path}: checkpoint contains a '
                'non-trailing execution error.'
            )
        failed = rows.pop()
        print(
            'Resume checkpoint found a failed final attempt; retrying '
            f'question {failed.get("id", "unknown")} without changing prior rows.'
        )

    # Every preserved row must be a completed, exact prefix of the selected
    # benchmark. This prevents accidental mixing after --start/--limit changes.
    for index, row in enumerate(rows, start=1):
        expected = _case_identity(selected[index - 1], index)
        for field, expected_value in expected.items():
            if str(row.get(field, '')) != expected_value:
                raise RuntimeError(
                    f'Refusing to resume {output_path}: saved question {index} '
                    f'field {field} does not match the active benchmark.'
                )
        if (
            'retrieval_evaluation' not in row
            or 'generation_evaluation' not in row
            or 'end_to_end_pass' not in row
        ):
            raise RuntimeError(
                f'Refusing to resume {output_path}: saved question {index} '
                'is not a completed evaluation row.'
            )

    is_complete = len(rows) == len(selected)
    if is_complete:
        print(
            f'Resume: {output_path.name} is already complete '
            f'({len(rows)}/{len(selected)}); no questions will be rerun.'
        )
    elif rows:
        print(
            f'Resume: preserving {len(rows)}/{len(selected)} completed '
            f'questions from {output_path.name}; continuing at question '
            f'{len(rows) + 1}.'
        )
    else:
        print(
            f'Resume: {output_path.name} contains no completed rows; '
            'starting from question 1.'
        )

    return rows, is_complete


def _run_once(
    *,
    args: argparse.Namespace,
    benchmark: dict[str, Any],
    selected: list[dict[str, Any]],
    output_path: Path,
    run_index: int,
    runs_requested: int,
    model_name: str,
) -> dict[str, Any]:
    strict_execution = True
    rows, is_complete = _load_resume_checkpoint(
        args=args,
        benchmark=benchmark,
        selected=selected,
        output_path=output_path,
        run_index=run_index,
        runs_requested=runs_requested,
        model_name=model_name,
    )
    if is_complete:
        # Rebuild from the preserved rows so summary/metadata reflect the
        # current runner schema while the model outputs themselves remain
        # untouched.
        output = build_output(
            args,
            benchmark,
            rows,
            run_index=run_index,
            runs_requested=runs_requested,
            model_name=model_name,
        )
        save_output(output_path, output)
        return output

    session = requests.Session()
    client = OpenAI()
    retrieve_url = args.base_url.rstrip('/') + '/retrieve'
    generate_url = args.base_url.rstrip('/') + '/generate?include_debug=false'

    start_offset = len(rows)
    for index, case in enumerate(
        selected[start_offset:],
        start=start_offset + 1,
    ):
        row: dict[str, Any] = _case_identity(case, index)

        retrieval: dict[str, Any] | None = None
        retrieval_elapsed: float | None = None
        generation: dict[str, Any] | None = None
        generation_elapsed: float | None = None

        try:
            retrieval, retrieval_elapsed = post_json(
                session,
                retrieve_url,
                {'question': row['question'], 'include_debug': False},
                args.timeout,
            )
            generation, generation_elapsed = post_json(
                session,
                generate_url,
                retrieval,
                args.timeout,
            )

            if strict_execution:
                validate_execution_integrity(
                    case_id=row['id'],
                    expected_behavior=row['expected_behavior'],
                    retrieval=retrieval,
                    generation=generation,
                    expected_model=get_settings().pipeline_llm_model,
                )

            retrieval_eval = evaluate_retrieval(
                case, retrieval, retrieval_elapsed
            )
            generation_eval = evaluate_generation(
                case,
                retrieval,
                generation,
                generation_elapsed,
                client,
                args.judge_model,
            )
            row.update({
                'retrieval_evaluation': retrieval_eval,
                'generation_evaluation': generation_eval,
                'end_to_end_pass': bool(
                    (
                        retrieval_eval['routing_correct']
                        and generation_eval['out_of_scope_response_correct']
                    )
                    if row['expected_behavior'] == 'abstain'
                    else (
                        retrieval_eval['routing_correct']
                        and retrieval_eval['article_recall_at_5'] == 1.0
                        and generation_eval['correctness_grade'] == 2
                        and generation_eval['faithfulness_grade'] >= 1
                        and generation_eval['citation_validity'] == 1.0
                        and generation_eval['citation_recall'] == 1.0
                    )
                ),
                'retrieval_output': retrieval,
                'generation_output': generation,
            })
        except ModelContractFailure as exc:
            retrieval_eval, generation_eval = (
                model_contract_failure_evaluations(
                    case,
                    stage=exc.stage,
                    retrieval=retrieval,
                    retrieval_elapsed=retrieval_elapsed,
                )
            )

            row.update({
                'model_contract_failure': True,
                'model_contract_stage': exc.stage,
                'model_contract_detail': exc.detail[:4000],
                'retrieval_evaluation': retrieval_eval,
                'generation_evaluation': generation_eval,
                'end_to_end_pass': False,
            })

            if retrieval is not None:
                row['retrieval_output'] = retrieval

            rows.append(row)

            save_output(
                output_path,
                build_output(
                    args,
                    benchmark,
                    rows,
                    run_index=run_index,
                    runs_requested=runs_requested,
                    model_name=model_name,
                ),
            )

            print_row(row, index, len(selected))

            if args.delay > 0:
                time.sleep(args.delay)

            if args.stop_on_error:
                break

            continue

        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
            row['end_to_end_pass'] = False
            rows.append(row)
            save_output(
                output_path,
                build_output(
                    args, benchmark, rows,
                    run_index=run_index,
                    runs_requested=runs_requested,
                    model_name=model_name,
                ),
            )
            print_row(row, index, len(selected))
            if strict_execution or args.stop_on_error:
                break
            continue

        rows.append(row)
        save_output(
            output_path,
            build_output(
                args, benchmark, rows,
                run_index=run_index,
                runs_requested=runs_requested,
                model_name=model_name,
            ),
        )
        print_row(row, index, len(selected))
        if args.delay > 0:
            time.sleep(args.delay)

    output = build_output(
        args,
        benchmark,
        rows,
        run_index=run_index,
        runs_requested=runs_requested,
        model_name=model_name,
    )
    save_output(output_path, output)
    return output


def main() -> int:
    args = parse_args()
    strict_execution = True

    benchmark = load_json(args.benchmark)
    if not benchmark.get('frozen'):
        raise ValueError('Benchmark must be frozen=true.')
    questions = benchmark.get('questions')
    if not isinstance(questions, list) or not questions:
        raise ValueError('Benchmark has no questions.')

    selected = questions[max(args.start - 1, 0):]
    if args.limit is not None:
        selected = selected[:max(args.limit, 0)]
    if not selected:
        raise ValueError('No questions selected.')

    settings = get_settings()
    model_name = str(args.model_name or settings.pipeline_llm_model)

    # An explicit --output is exactly one run. This keeps
    # the previous smoke-test commands valid.
    runs_requested = 1 if args.output is not None else int(args.runs)
    if runs_requested < 1:
        raise ValueError('--runs must be at least 1.')

    if args.output is not None:
        output_root = args.output.parent
    else:
        experiment_slug = safe_slug(
            f'{settings.pipeline_llm_provider}__{model_name}'
        )
        output_root = args.output_dir or (
            DEFAULT_RESULTS_DIR / experiment_slug
        )
    output_root.mkdir(parents=True, exist_ok=True)

    run_outputs: list[dict[str, Any]] = []
    run_paths: list[Path] = []

    for run_index in range(1, runs_requested + 1):
        output_path = (
            args.output
            if args.output is not None
            else output_root / f'run-{run_index}.json'
        )
        assert output_path is not None

        print(
            f'\n=== {settings.pipeline_llm_provider}/{settings.pipeline_llm_model} '
            f'| run {run_index}/{runs_requested} '
            f'| questions={len(selected)} ==='
        )
        output = _run_once(
            args=args,
            benchmark=benchmark,
            selected=selected,
            output_path=output_path,
            run_index=run_index,
            runs_requested=runs_requested,
            model_name=model_name,
        )
        run_outputs.append(output)
        run_paths.append(output_path)

        print('\nRun summary')
        print(json.dumps(output['summary'], ensure_ascii=False, indent=2))
        print(f'Saved run file: {output_path}')

        if output['summary']['questions_failed_to_run'] != 0:
            print(
                '\nExecution failure detected. Remaining repetitions are not '
                'started because a final aggregate must contain only complete runs.'
            )
            break

    # Old explicit --output behavior: one result file and normal exit code.
    if args.output is not None:
        return (
            0
            if run_outputs
            and run_outputs[0]['summary']['questions_failed_to_run'] == 0
            else 1
        )

    aggregate = build_aggregate_output(
        args=args,
        benchmark=benchmark,
        run_outputs=run_outputs,
        run_paths=run_paths,
        runs_requested=runs_requested,
        selected_count=len(selected),
        model_name=model_name,
    )
    final_path = output_root / 'final-summary.json'
    save_output(final_path, aggregate)

    print('\n=== FINAL AGGREGATE ===')
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    print(f'\nSaved final summary: {final_path}')

    return 0 if aggregate['status'] == 'valid' else 1


if __name__ == '__main__':
    raise SystemExit(main())