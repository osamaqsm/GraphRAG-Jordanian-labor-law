# Fresh Final 40 Benchmark Validation Report

## Package
- Benchmark: `jordan_labor_law_fresh_final_40_v2_3.json`
- Version: `1.0.0`
- Created: `2026-08-13`
- Frozen: `true`
- Pipeline freeze label: `multi-article-v2.3`
- SHA-256: `547d4636800043e14cd9254b566cbcf6aabfc464f046c798011ea3edd72077c7`

## Counts
- Total: 40
- Retrieve: 36
- Abstain: 4
- Clarify: 0

## Question styles
- colloquial: 5
- multi_article: 7
- numerical: 5
- out_of_scope: 4
- paraphrased: 8
- straightforward: 8
- typo: 3

## Difficulty
- easy: 6
- medium: 17
- hard: 17

## Gold safeguards
- Unique gold articles: 60
- Gold overlap with preceding V2.3 diagnostic 40: 0
- Exact normalized question overlap with six historical benchmark artifacts: 0
- Duplicate IDs: 0
- Duplicate normalized questions inside this benchmark: 0
- Gold article bounds check (1-142): PASS
- Required-citation equality check: PASS

## Gold articles
1, 5, 6, 7, 8, 9, 10, 11, 19, 33, 34, 37, 38, 45, 49, 50, 51, 52, 53, 54, 56, 58, 63, 64, 69, 73, 74, 75, 77, 78, 79, 80, 81, 83, 84, 85, 89, 91, 97, 98, 100, 102, 103, 105, 106, 108, 109, 117, 118, 123, 125, 127, 128, 130, 131, 135, 136, 137, 138, 139

## Multi-article cases
- FH40-30: [5, 6, 7, 8, 9]
- FH40-31: [37, 38]
- FH40-32: [49, 50, 51, 52, 53]
- FH40-33: [73, 74, 75, 77]
- FH40-34: [78, 79, 80, 84, 85]
- FH40-35: [98, 100, 102, 103, 106]
- FH40-36: [125, 127, 128, 130, 131]

## Source
- KG SHA-256: `82960a99419a0ee739a0154c0c4173f0e4caf7172f885211f3c26dbf5170ab4b`
- KG article coverage: 1-142
- Preceding diagnostic benchmark SHA-256: `a356c5a16c20c382aa377294105062152adbf5132dbea82801ebcd9ca2159739`

## Methodological note
The V2.3 diagnostic benchmark was used during architecture refinement, so it is not the final unseen estimate.
This new package was created after the V2.3 freeze. Its gold articles have zero overlap with that diagnostic
benchmark. Some statutes may have appeared in older historical development benchmarks; therefore the defensible
claim is a fresh question holdout with gold disjointness from the immediately preceding tuning benchmark, not
globally never-before-seen statutory provisions.

## First-run rule
Do not modify this benchmark or the frozen pipeline after observing the first official model result.
