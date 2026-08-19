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

DEFAULT_BENCHMARK = Path('/app/data/benchmarks/model_comparison_benchmark_40.json')
DEFAULT_RESULTS_DIR = Path('/app/data/model_evaluations')
ARABIC_RE = re.compile(r'[\u0600-\u06FF]')
LATIN_RE = re.compile(r'[A-Za-z]')
CITATION_RE = re.compile(r'\[\s*المادة\s+([0-9٠-٩۰-۹]+)\s*\]')
DIGIT_TRANS = str.maketrans('٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹', '01234567890123456789')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Run the complete Jordanian Labor Law pipeline on the fixed '
            '40-question benchmark and write one evaluation JSON file.'
        )
    )
    parser.add_argument('--model-name', required=True, help='Label of the evaluated model.')
    parser.add_argument('--benchmark', type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--base-url', default='http://localhost:8000')
    parser.add_argument('--judge-model', default=os.getenv('EVALUATION_JUDGE_MODEL', 'gpt-5.4-mini'))
    parser.add_argument('--timeout', type=float, default=240.0)
    parser.add_argument('--delay', type=float, default=0.0)
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--stop-on-error', action='store_true')
    parser.add_argument(
        '--allow-llm-fallback',
        action='store_true',
        help=(
            'Diagnostic only. Allow deterministic fallback when an LLM stage '
            'fails. Final model-comparison runs should NOT use this flag.'
        ),
    )
    return parser.parse_args()


class ExecutionIntegrityError(RuntimeError):
    """Raised when a model-comparison run stops using the configured LLM."""


def truthy(value: Any) -> bool:
    return str(value or '').strip().lower() not in {
        '', '0', 'false', 'no', 'off'
    }


def validate_execution_integrity(
    *,
    case_id: str,
    expected_behavior: str,
    retrieval: dict[str, Any],
    generation: dict[str, Any],
    expected_model: str,
) -> None:
    """Fail closed when a benchmark case silently leaves the LLM path.

    The planner flag is observable in retrieval.v1. Reranker/provider failures
    are made fatal by PIPELINE_STRICT_EVALUATION inside the API components.
    Generation provider failures are also fatal in strict mode, while genuine
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


def post_json(session: requests.Session, url: str, payload: dict[str, Any], timeout: float) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    response = session.post(url, json=payload, timeout=timeout)
    elapsed = time.perf_counter() - started
    response.raise_for_status()
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

def build_output(args: argparse.Namespace, benchmark: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    settings = get_settings()

    return {
        'schema_version': 'full-pipeline-evaluation.v2',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'model_name': args.model_name,
        'judge_model': args.judge_model,
        'execution_integrity': {
            'strict_llm_execution': not args.allow_llm_fallback,
            'pipeline_strict_evaluation_env': truthy(
                os.getenv('PIPELINE_STRICT_EVALUATION', 'false')
            ),
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

def main() -> int:
    args = parse_args()
    strict_execution = not args.allow_llm_fallback

    if strict_execution and not truthy(
        os.getenv('PIPELINE_STRICT_EVALUATION', 'false')
    ):
        raise RuntimeError(
            'Final model-comparison runs require '
            'PIPELINE_STRICT_EVALUATION=true in the API container. '
            'This prevents planner/reranker/generator provider failures from '
            'silently falling back to deterministic behavior.'
        )

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

    output_path = args.output or (DEFAULT_RESULTS_DIR / f'{safe_slug(args.model_name)}.json')
    session = requests.Session()
    client = OpenAI()
    rows: list[dict[str, Any]] = []
    retrieve_url = args.base_url.rstrip('/') + '/retrieve'
    generate_url = args.base_url.rstrip('/') + '/generate?include_debug=false'

    for index, case in enumerate(selected, start=1):
        row: dict[str, Any] = {
            'id': str(case.get('id', f'question_{index}')),
            'question': str(case.get('question', '')),
            'test_type': str(case.get('test_type', 'unknown')),
            'category': str(case.get('category', 'unknown')),
            'difficulty': str(case.get('difficulty', 'unknown')),
            'expected_behavior': str(case.get('expected_behavior', 'retrieve')),
        }
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

            retrieval_eval = evaluate_retrieval(case, retrieval, retrieval_elapsed)
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
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
            row['end_to_end_pass'] = False
            rows.append(row)
            save_output(output_path, build_output(args, benchmark, rows))
            print_row(row, index, len(selected))
            if strict_execution or args.stop_on_error:
                break
            continue

        rows.append(row)
        save_output(output_path, build_output(args, benchmark, rows))
        print_row(row, index, len(selected))
        if args.delay > 0:
            time.sleep(args.delay)

    output = build_output(args, benchmark, rows)
    save_output(output_path, output)
    print('\nSummary')
    print(json.dumps(output['summary'], ensure_ascii=False, indent=2))
    print(f'\nSaved one final file: {output_path}')
    return 0 if output['summary']['questions_failed_to_run'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())