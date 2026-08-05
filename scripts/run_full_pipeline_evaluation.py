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
    parser.add_argument('--judge-model', default=os.getenv('EVALUATION_JUDGE_MODEL', 'gpt-4o-mini'))
    parser.add_argument('--timeout', type=float, default=240.0)
    parser.add_argument('--delay', type=float, default=0.0)
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--stop-on-error', action='store_true')
    return parser.parse_args()


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


def evaluate_retrieval(case: dict[str, Any], retrieval: dict[str, Any], elapsed: float) -> dict[str, Any]:
    expected_behavior = str(case.get('expected_behavior', 'retrieve'))
    actual_behavior = get_behavior(retrieval)
    actual = actual_article_numbers(retrieval)
    required = [int(x) for x in case.get('required_articles', [])]
    acceptable = [int(x) for x in case.get('acceptable_articles', required)]

    if expected_behavior == 'retrieve':
        required_set = set(required)
        acceptable_set = set(acceptable)
        actual_set = set(actual)
        hit1 = bool(actual and actual[0] in acceptable_set)
        hit3 = bool(set(actual[:3]) & required_set)
        recall = len(actual_set & required_set) / len(required_set) if required_set else 1.0
        precision = len(actual_set & acceptable_set) / len(actual_set) if actual_set else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        exact_set = actual_set == required_set
        reciprocal_rank = 0.0
        for rank, number in enumerate(actual, start=1):
            if number in required_set:
                reciprocal_rank = 1.0 / rank
                break
        passed = actual_behavior == 'retrieve' and hit1 and recall == 1.0
    else:
        hit1 = hit3 = False
        recall = precision = f1 = reciprocal_rank = 0.0
        exact_set = not actual
        passed = actual_behavior == expected_behavior and not actual

    return {
        'expected_behavior': expected_behavior,
        'actual_behavior': actual_behavior,
        'required_articles': required,
        'acceptable_articles': acceptable,
        'actual_articles': actual,
        'passed': passed,
        'hit_at_1': hit1,
        'hit_at_3': hit3,
        'reciprocal_rank': round(reciprocal_rank, 6),
        'exact_set': exact_set,
        'required_article_recall': round(recall, 6),
        'strict_precision': round(precision, 6),
        'strict_f1': round(f1, 6),
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
    required_facts = [str(x) for x in case.get('required_facts', [])]
    forbidden_claims = [str(x) for x in case.get('forbidden_claims', [])]
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
            'required_fact_supported': {
                'type': 'array',
                'items': {'type': 'boolean'},
            },
            'forbidden_claim_present': {
                'type': 'array',
                'items': {'type': 'boolean'},
            },
            'answer_correctness_score': {'type': 'number', 'minimum': 0, 'maximum': 1},
            'faithfulness_score': {'type': 'number', 'minimum': 0, 'maximum': 1},
            'rationale_ar': {'type': 'string'},
        },
        'required': [
            'required_fact_supported',
            'forbidden_claim_present',
            'answer_correctness_score',
            'faithfulness_score',
            'rationale_ar',
        ],
    }
    payload = {
        'question': case.get('question'),
        'required_facts': required_facts,
        'forbidden_claims': forbidden_claims,
        'retrieved_legal_evidence': evidence,
        'generated_answer': generation.get('answer_ar', ''),
        'generated_citations': extract_citations(generation),
    }
    instructions = (
        'You are a strict evaluator of Arabic legal question answering. '
        'Use only the supplied retrieved legal evidence and rubric. '
        'For each required fact, return true only if the generated answer states it correctly. '
        'For each forbidden claim, return true only if the answer makes that claim. '
        'Answer correctness measures correctness and completeness against required facts. '
        'Faithfulness measures whether every substantive legal claim is supported by the supplied evidence. '
        'Do not reward verbosity. Do not use outside legal knowledge. '
        'The two boolean arrays must have exactly the same lengths and order as their input arrays.'
    )
    response = client.responses.create(
        model=judge_model,
        instructions=instructions,
        input=json.dumps(payload, ensure_ascii=False, separators=(',', ':')),
        text={
            'format': {
                'type': 'json_schema',
                'name': 'legal_generation_evaluation',
                'schema': schema,
                'strict': True,
            }
        },
        store=False,
    )
    result = json.loads(response_text(response))
    facts = list(result.get('required_fact_supported', []))
    forbidden = list(result.get('forbidden_claim_present', []))
    if len(facts) != len(required_facts):
        raise ValueError('Judge returned wrong required_fact_supported length.')
    if len(forbidden) != len(forbidden_claims):
        raise ValueError('Judge returned wrong forbidden_claim_present length.')
    return {
        'required_fact_supported': [bool(x) for x in facts],
        'forbidden_claim_present': [bool(x) for x in forbidden],
        'answer_correctness_score': round(float(result['answer_correctness_score']), 6),
        'faithfulness_score': round(float(result['faithfulness_score']), 6),
        'rationale_ar': str(result.get('rationale_ar', '')),
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
    required_citations = [int(x) for x in case.get('required_citations', case.get('required_articles', []))]
    retrieved_numbers = actual_article_numbers(retrieval)

    citations_set = set(citations)
    required_set = set(required_citations)
    retrieved_set = set(retrieved_numbers)
    citation_validity = (
        len(citations_set & retrieved_set) / len(citations_set)
        if citations_set else (1.0 if expected_behavior != 'retrieve' else 0.0)
    )
    citation_precision = (
        len(citations_set & required_set) / len(citations_set)
        if citations_set else (1.0 if not required_set else 0.0)
    )
    citation_recall = (
        len(citations_set & required_set) / len(required_set)
        if required_set else (1.0 if not citations_set else 0.0)
    )
    citation_exact_set = citations_set == required_set
    generation_success = bool(answer)

    if expected_behavior == 'abstain':
        arabic_only = is_arabic_only(answer)
        status_correct = status == 'out_of_scope'
        strict_pass = generation_success and arabic_only and status_correct and not citations
        return {
            'status': status,
            'answer_ar': answer,
            'cited_article_numbers': citations,
            'generation_success': generation_success,
            'status_correct': status_correct,
            'arabic_only': arabic_only,
            'citation_validity': round(citation_validity, 6),
            'citation_precision': round(citation_precision, 6),
            'citation_recall': round(citation_recall, 6),
            'citation_exact_set': citation_exact_set,
            'required_fact_coverage': 1.0 if strict_pass else 0.0,
            'answer_correctness': 1.0 if strict_pass else 0.0,
            'faithfulness': 1.0 if strict_pass else 0.0,
            'forbidden_claim_rate': 0.0,
            'strict_pass': strict_pass,
            'elapsed_seconds': round(elapsed, 3),
            'judge': None,
        }

    judge = judge_generation(client, judge_model, case, retrieval, generation)
    fact_flags = judge['required_fact_supported']
    forbidden_flags = judge['forbidden_claim_present']
    fact_coverage = sum(fact_flags) / len(fact_flags) if fact_flags else 1.0
    forbidden_rate = sum(forbidden_flags) / len(forbidden_flags) if forbidden_flags else 0.0
    status_correct = status in {'answered', 'success', 'grounded_answer'} or bool(generation.get('grounded'))
    strict_pass = (
        generation_success
        and status_correct
        and fact_coverage == 1.0
        and forbidden_rate == 0.0
        and judge['faithfulness_score'] >= 0.999
        and citation_recall == 1.0
        and citation_validity == 1.0
    )
    return {
        'status': status,
        'answer_ar': answer,
        'cited_article_numbers': citations,
        'generation_success': generation_success,
        'status_correct': status_correct,
        'arabic_only': is_arabic_only(answer),
        'citation_validity': round(citation_validity, 6),
        'citation_precision': round(citation_precision, 6),
        'citation_recall': round(citation_recall, 6),
        'citation_exact_set': citation_exact_set,
        'required_fact_coverage': round(fact_coverage, 6),
        'answer_correctness': judge['answer_correctness_score'],
        'faithfulness': judge['faithfulness_score'],
        'forbidden_claim_rate': round(forbidden_rate, 6),
        'strict_pass': strict_pass,
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
        'end_to_end_strict_accuracy': mean(completed, ('end_to_end_pass',)),
        'retrieval': {
            'overall_accuracy': mean(completed, ('retrieval_evaluation', 'passed')),
            'retrieve_pass_rate': mean(retrieve, ('retrieval_evaluation', 'passed')),
            'hit_at_1': mean(retrieve, ('retrieval_evaluation', 'hit_at_1')),
            'hit_at_3': mean(retrieve, ('retrieval_evaluation', 'hit_at_3')),
            'mean_reciprocal_rank': mean(retrieve, ('retrieval_evaluation', 'reciprocal_rank')),
            'exact_set_accuracy': mean(retrieve, ('retrieval_evaluation', 'exact_set')),
            'required_article_recall': mean(retrieve, ('retrieval_evaluation', 'required_article_recall')),
            'mean_strict_precision': mean(retrieve, ('retrieval_evaluation', 'strict_precision')),
            'mean_strict_f1': mean(retrieve, ('retrieval_evaluation', 'strict_f1')),
            'out_of_scope_abstention_accuracy': mean(abstain, ('retrieval_evaluation', 'passed')),
            'mean_latency_seconds': mean(completed, ('retrieval_evaluation', 'elapsed_seconds')),
        },
        'generation': {
            'strict_accuracy': mean(completed, ('generation_evaluation', 'strict_pass')),
            'generation_success_rate': mean(completed, ('generation_evaluation', 'generation_success')),
            'status_accuracy': mean(completed, ('generation_evaluation', 'status_correct')),
            'answer_correctness': mean(retrieve, ('generation_evaluation', 'answer_correctness')),
            'required_fact_coverage': mean(retrieve, ('generation_evaluation', 'required_fact_coverage')),
            'faithfulness': mean(retrieve, ('generation_evaluation', 'faithfulness')),
            'citation_validity': mean(retrieve, ('generation_evaluation', 'citation_validity')),
            'citation_precision': mean(retrieve, ('generation_evaluation', 'citation_precision')),
            'citation_recall': mean(retrieve, ('generation_evaluation', 'citation_recall')),
            'citation_exact_set_accuracy': mean(retrieve, ('generation_evaluation', 'citation_exact_set')),
            'forbidden_claim_rate': mean(retrieve, ('generation_evaluation', 'forbidden_claim_rate')),
            'arabic_only_out_of_scope_accuracy': mean(abstain, ('generation_evaluation', 'arabic_only')),
            'mean_latency_seconds': mean(completed, ('generation_evaluation', 'elapsed_seconds')),
        },
        'by_test_type': {
            test_type: {
                'count': len(group),
                'end_to_end_accuracy': mean(group, ('end_to_end_pass',)),
                'retrieval_accuracy': mean(group, ('retrieval_evaluation', 'passed')),
                'generation_accuracy': mean(group, ('generation_evaluation', 'strict_pass')),
            }
            for test_type, group in sorted(by_type.items())
        },
    }


def build_output(args: argparse.Namespace, benchmark: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'schema_version': 'full-pipeline-evaluation.v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'model_name': args.model_name,
        'judge_model': args.judge_model,
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


def print_row(row: dict[str, Any], index: int, total: int) -> None:
    if row.get('error'):
        print(f'[{index:02d}/{total:02d}] {row["id"]} ERROR | {row["error"]}')
        return
    r = row['retrieval_evaluation']
    g = row['generation_evaluation']
    label = 'PASS' if row['end_to_end_pass'] else 'FAIL'
    print(
        f'[{index:02d}/{total:02d}] {row["id"]} {label} '
        f'| R={int(bool(r["passed"]))} '
        f'| G={int(bool(g["strict_pass"]))} '
        f'| Hit@1={int(bool(r["hit_at_1"])) if row["expected_behavior"] == "retrieve" else "-"} '
        f'| Facts={g["required_fact_coverage"]:.2f} '
        f'| Faith={g["faithfulness"]:.2f}'
    )


def main() -> int:
    args = parse_args()
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
                'end_to_end_pass': bool(retrieval_eval['passed'] and generation_eval['strict_pass']),
                'retrieval_output': retrieval,
                'generation_output': generation,
            })
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {exc}'
            row['end_to_end_pass'] = False
            rows.append(row)
            save_output(output_path, build_output(args, benchmark, rows))
            print_row(row, index, len(selected))
            if args.stop_on_error:
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
