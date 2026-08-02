from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

from rdflib import Graph, Namespace, RDFS

DEFAULT_BENCHMARK = Path('/app/data/benchmarks/retrieval_benchmark_unseen_30_final.json')
DEFAULT_TTL = Path('/app/data/jordan_labor_law_full_knowledge_graph.ttl')
DEFAULT_DEV20 = Path('/app/data/benchmarks/retrieval_benchmark_20.json')
DEFAULT_PREVIOUS50 = Path('/app/data/benchmarks/retrieval_benchmark_unseen_50.json')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_arabic(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r'[إأآٱ]', 'ا', text)
    text = text.replace('ى', 'ي').replace('ة', 'ه').replace('ؤ', 'و').replace('ئ', 'ي')
    text = re.sub(r'[\u064b-\u065f\u0670\u06d6-\u06ed]', '', text)
    text = re.sub(r'[^\u0600-\u06ff0-9a-z]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def load_questions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding='utf-8-sig'))
    return value.get('questions', [])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument('--ttl', type=Path, default=DEFAULT_TTL)
    parser.add_argument('--dev20', type=Path, default=DEFAULT_DEV20)
    parser.add_argument('--previous50', type=Path, default=DEFAULT_PREVIOUS50)
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding='utf-8-sig'))
    questions = benchmark['questions']
    assert benchmark.get('frozen') is True
    assert len(questions) == 30
    assert len({q['id'] for q in questions}) == 30
    assert len({normalize_arabic(q['question']) for q in questions}) == 30

    distribution = Counter(q['test_type'] for q in questions)
    assert dict(distribution) == benchmark['distribution'], (distribution, benchmark['distribution'])
    assert dict(distribution) == {
        'straightforward': 5,
        'paraphrased': 5,
        'typo': 3,
        'colloquial': 3,
        'numerical': 3,
        'multi_article': 5,
        'ambiguous': 3,
        'out_of_scope': 3,
    }

    graph = Graph(); graph.parse(args.ttl)
    ns = Namespace('http://example.org/jordan-labor-law#')
    concept_names = {
        str(subject).split('#')[-1]
        for subject in graph.subjects(RDFS.label, None)
        if str(subject).startswith(str(ns))
    }

    retrieve = clarify = abstain = 0
    current_gold: set[int] = set()
    for case in questions:
        behavior = case['expected_behavior']
        if behavior == 'retrieve':
            retrieve += 1
            required = [int(v) for v in case['required_articles']]
            acceptable = [int(v) for v in case.get('acceptable_articles', required)]
            assert case['primary_article'] in required
            assert 1 <= len(required) <= 3
            assert len(required) == len(set(required))
            current_gold.update(required)
            for article_number in set(required + acceptable):
                article_uri = ns[f'article_{article_number}']
                comments = list(graph.objects(article_uri, RDFS.comment))
                assert comments and str(comments[0]).strip(), (case['id'], article_number)
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

    assert (retrieve, clarify, abstain) == (24, 3, 3)

    previous_questions = load_questions(args.dev20) + load_questions(args.previous50)
    previous_texts = [normalize_arabic(q['question']) for q in previous_questions]
    current_texts = [normalize_arabic(q['question']) for q in questions]
    assert not (set(previous_texts) & set(current_texts)), 'Exact normalized question overlap found.'

    max_similarity = 0.0
    closest_pair = None
    for current in current_texts:
        for previous in previous_texts:
            score = SequenceMatcher(None, current, previous).ratio()
            if score > max_similarity:
                max_similarity = score
                closest_pair = (current, previous)
    assert max_similarity < 0.82, (max_similarity, closest_pair)

    previous_gold: set[int] = set()
    for case in previous_questions:
        previous_gold.update(int(v) for v in case.get('required_articles', []))
        previous_gold.update(int(v) for v in case.get('acceptable_articles', []))
    overlap = sorted(current_gold & previous_gold)
    assert not overlap, f'Gold article overlap with previous benchmarks: {overlap}'

    print('Final unseen 30 benchmark validation passed.')
    print(f'Questions: {len(questions)}')
    print(f'Retrieve / clarify / abstain: {retrieve} / {clarify} / {abstain}')
    print(f'Distribution: {dict(distribution)}')
    print(f'Unique new gold articles: {len(current_gold)}')
    print(f'Max normalized text similarity to prior benchmarks: {max_similarity:.3f}')
    print(f'SHA-256: {sha256(args.benchmark)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
