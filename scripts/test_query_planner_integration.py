from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = Path('/tmp/query_planner_integration_results.json')

CASES = [
    {
        'id': 'Q1',
        'question': 'ما شروط عقد التدريب ومتى يجوز إنهاؤه إذا أصبح خطراً على صحة المتدرب؟',
        'expected_behavior': 'retrieve',
    },
    {
        'id': 'Q2',
        'question': 'كم يستطيع صاحب العمل اقتطاعه من الأجر لاسترداد سلفة؟',
        'expected_behavior': 'retrieve',
    },
    {
        'id': 'Q3',
        'question': 'النقابة اتخذت قراراً ضدي، هل قرارها قانوني؟',
        'expected_behavior': 'clarify',
    },
    {
        'id': 'Q4',
        'question': 'صار نزاع عمالي جماعي، شو الخطوة القانونية الجاية؟',
        'expected_behavior': 'clarify',
    },
    {
        'id': 'Q5',
        'question': 'اشتريت هاتفاً وظهر فيه عيب، هل أستطيع استبداله؟',
        'expected_behavior': 'abstain',
    },
    {
        'id': 'Q6',
        'question': 'الجامعة أوقفت تسجيلي بسبب المعدل، كيف أعترض؟',
        'expected_behavior': 'abstain',
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Run Stage 7.7-B query-planner end-to-end integration checks.'
    )
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def extract_result_json(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != '{':
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and 'retrieval' in value and 'status' in value:
            return value
    raise ValueError('Could not find retrieval JSON in command output.')


def ordered_unique(values: list[int]) -> list[int]:
    return list(dict.fromkeys(values))


def article_numbers(result: dict[str, Any]) -> list[int]:
    diagnostics = result.get('diagnostics', {})
    numbers = diagnostics.get('top_article_numbers')
    if isinstance(numbers, list):
        return ordered_unique([int(value) for value in numbers if value is not None])

    hits = result.get('retrieval', {}).get('article_hits', [])
    return ordered_unique([
        int(hit['article_number'])
        for hit in hits
        if hit.get('article_number') is not None
    ])


def concept_names(result: dict[str, Any]) -> list[str]:
    retrieval = result.get('retrieval', {})
    names: list[str] = []
    for key in ('concept_hits', 'expanded_concept_hits'):
        for hit in retrieval.get(key, []):
            name = str(hit.get('local_name', '')).strip()
            if name:
                names.append(name)
    return list(dict.fromkeys(names))


def run_question(question: str) -> tuple[dict[str, Any], float, str]:
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    command = [sys.executable, '-m', 'scripts.test_retrieval', question]

    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd='/app',
        env=env,
        text=True,
        encoding='utf-8',
        errors='replace',
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - started

    if completed.returncode != 0:
        raise RuntimeError(
            f'scripts.test_retrieval exited with {completed.returncode}.\n'
            f'STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}'
        )

    return extract_result_json(completed.stdout), elapsed, completed.stderr


def evaluate(case: dict[str, str], raw: dict[str, Any], elapsed: float) -> dict[str, Any]:
    articles = article_numbers(raw)
    expected = case['expected_behavior']

    if expected == 'retrieve':
        passed = len(articles) > 0
        observed = 'retrieved' if articles else 'no_retrieval'
    else:
        passed = len(articles) == 0
        observed = 'safe_no_retrieval' if not articles else 'retrieved'

    return {
        'id': case['id'],
        'question': case['question'],
        'expected_behavior': expected,
        'observed_behavior': observed,
        'articles': articles,
        'concepts': concept_names(raw),
        'article_count': len(articles),
        'pass': passed,
        'elapsed_seconds': round(elapsed, 3),
    }


def main() -> int:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for case in CASES:
        try:
            raw, elapsed, stderr = run_question(case['question'])
            row = evaluate(case, raw, elapsed)
            if stderr.strip():
                row['stderr'] = stderr.strip()
            rows.append(row)
            print(
                f"{case['id']}: expected={case['expected_behavior']} "
                f"observed={row['observed_behavior']} articles={row['articles']} "
                f"pass={row['pass']}"
            )
        except Exception as exc:  # noqa: BLE001 - test runner records failures
            errors.append({'id': case['id'], 'error': str(exc)})
            rows.append({
                'id': case['id'],
                'question': case['question'],
                'expected_behavior': case['expected_behavior'],
                'observed_behavior': 'error',
                'articles': [],
                'concepts': [],
                'article_count': 0,
                'pass': False,
                'error': str(exc),
            })
            print(f"{case['id']}: ERROR {exc}")

    counts = Counter(row['observed_behavior'] for row in rows)
    passed = sum(bool(row.get('pass')) for row in rows)
    output = {
        'test_name': 'Stage 7.7-B Query Planner Integration Test',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'planner_enabled': os.getenv('OPENAI_QUERY_PLANNER_ENABLED', '').lower() == 'true',
        'planner_model': os.getenv('OPENAI_QUERY_PLANNER_MODEL') or os.getenv('OPENAI_CHAT_MODEL'),
        'reasoning_effort': os.getenv('OPENAI_QUERY_PLANNER_REASONING_EFFORT', 'low'),
        'route_confidence': float(os.getenv('OPENAI_QUERY_PLANNER_ROUTE_CONFIDENCE', '0.80')),
        'retrieve_override_confidence': float(
            os.getenv('OPENAI_QUERY_PLANNER_RETRIEVE_OVERRIDE_CONFIDENCE', '0.90')
        ),
        'summary': {
            'questions_requested': len(CASES),
            'questions_completed': len(rows),
            'passed': passed,
            'accuracy': round(passed / len(CASES), 6),
            'errors': len(errors),
            'observed_behavior_counts': dict(counts),
        },
        'results': rows,
        'errors': errors,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(f'JSON result saved to {args.output}')
    return 0 if passed == len(CASES) and not errors else 1


if __name__ == '__main__':
    raise SystemExit(main())