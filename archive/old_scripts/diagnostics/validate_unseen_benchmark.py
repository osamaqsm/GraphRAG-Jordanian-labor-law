from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from rdflib import Graph, Namespace, RDFS

DEFAULT_BENCHMARK = Path('/app/data/benchmarks/retrieval_benchmark_unseen_50.json')
DEFAULT_TTL = Path('/app/data/jordan_labor_law_full_knowledge_graph.ttl')
DEFAULT_DEV = Path('/app/data/benchmarks/retrieval_benchmark_20.json')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument('--ttl', type=Path, default=DEFAULT_TTL)
    parser.add_argument('--dev-benchmark', type=Path, default=DEFAULT_DEV)
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding='utf-8-sig'))
    questions = benchmark['questions']
    assert benchmark.get('frozen') is True
    assert len(questions) == 50
    assert len({q['id'] for q in questions}) == 50
    assert len({q['question'].strip() for q in questions}) == 50

    distribution = Counter(q['test_type'] for q in questions)
    assert dict(distribution) == benchmark['distribution'], (distribution, benchmark['distribution'])

    graph = Graph(); graph.parse(args.ttl)
    ns = Namespace('http://example.org/jordan-labor-law#')
    concept_names = {
        str(subject).split('#')[-1]
        for subject in graph.subjects(RDFS.label, None)
        if str(subject).startswith(str(ns))
    }

    retrieve = clarify = abstain = 0
    for case in questions:
        behavior = case['expected_behavior']
        if behavior == 'retrieve':
            retrieve += 1
            required = case['required_articles']
            assert case['primary_article'] in required
            for article_number in set(required + case.get('acceptable_articles', [])):
                article_uri = ns[f'article_{article_number}']
                assert any(graph.objects(article_uri, RDFS.comment)), (case['id'], article_number)
            for concept in case.get('expected_concepts_any', []):
                assert concept in concept_names, (case['id'], concept)
        elif behavior == 'clarify':
            clarify += 1
            assert 'primary_article' not in case
        elif behavior == 'abstain':
            abstain += 1
            assert 'primary_article' not in case
        else:
            raise AssertionError((case['id'], behavior))

    assert (retrieve, clarify, abstain) == (40, 5, 5)

    if args.dev_benchmark.exists():
        dev = json.loads(args.dev_benchmark.read_text(encoding='utf-8-sig'))
        dev_questions = {q['question'].strip() for q in dev['questions']}
        unseen_questions = {q['question'].strip() for q in questions}
        assert not (dev_questions & unseen_questions)

    print('Unseen benchmark validation passed.')
    print(f'Questions: {len(questions)}')
    print(f'Retrieve / clarify / abstain: {retrieve} / {clarify} / {abstain}')
    print(f'Distribution: {dict(distribution)}')
    print(f'SHA-256: {sha256(args.benchmark)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
