# Stage 7.6-A — Scope and Ambiguity Gate

This patch is the first correction after the frozen unseen-50 baseline.

Changes:
- Adds a conservative pre-retrieval route: retrieve / clarify / abstain.
- Returns no article candidates for ambiguous or out-of-scope questions.
- Runs deterministic full-catalog candidate recovery for all questions,
  not only the 20 recognized development issue profiles.
- Changes unrecognized single-issue questions from a default of three
  final articles to one final article.
- Does not add or hard-code article numbers.

Files:
- app/legal_question_analysis.py
- app/retrieval_service.py
- scripts/test_stage_7_6a.py

Important:
The unseen-50 set has now been inspected and is a diagnostic/development set.
A new untouched benchmark will be required after Stage 7.6 tuning.
