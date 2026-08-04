# Stage 8-A — Frozen Retrieval Boundary

This patch creates a retrieval-only interface without adding answer generation.
It does not change article scoring, candidate retrieval, reranking, query-planner
thresholds, or benchmark data.

## Files

- `app/retrieval_contract.py`: versioned `retrieval.v1` JSON contract.
- `app/retrieval_pipeline.py`: routing + embedding + current RetrievalService.
- `app/retrieval_api.py`: `POST /retrieve` endpoint.
- `app/retrieval_service.py`: current file with one backward-compatible optional
  `analysis` parameter. Existing callers behave exactly as before.
- `scripts/retrieve.py`: retrieval-only CLI and JSON saver.
- `scripts/patch_main_stage_8a.py`: safely registers `/retrieve` in `app/main.py`.
- `scripts/test_stage_8a.py`: offline boundary checks.

## Install

Take a snapshot first, then copy the files into the matching project folders.
Run `python -m scripts.patch_main_stage_8a` once to register the route.

## Retrieval-only CLI

```powershell
python -m scripts.retrieve `
    "كم يستطيع صاحب العمل اقتطاعه من الأجر لاسترداد سلفة؟" `
    --output .\data\retrieval_results\loan_deduction.json `
    --debug
```

The saved JSON can later be passed to a generation-only command without
running retrieval again.

## API

`POST /retrieve`

```json
{
  "question": "كم يستطيع صاحب العمل اقتطاعه من الأجر لاسترداد سلفة؟",
  "include_debug": false
}
```

The response has `schema_version=retrieval.v1` and deliberately has no answer
or generation field.
