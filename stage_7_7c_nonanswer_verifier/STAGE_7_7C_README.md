# Stage 7.7-C — Non-answer verification

This patch addresses two findings from the Stage 7.7-B diagnostic:

1. False clarifications can ask the user to provide the legal conclusion
   (for example, whether a right is waivable) rather than a missing fact.
2. The same prompt can produce different routes across repeated calls.

The patch changes only `app/legal_query_planner.py` and adds two test scripts.

## Behavior

- The normal planner still makes one call.
- A second verification call occurs only when the first decision is
  `clarify` or `abstain`.
- The verifier challenges the proposed non-answer.
- A legal conclusion is not treated as a missing fact.
- Genuine ambiguities and out-of-scope questions remain non-answers.
- Verification can be disabled independently.
- Multiple retrieval queries may map to one governing article.

## Environment variables

Optional variables:

```env
OPENAI_QUERY_PLANNER_VERIFY_NON_ANSWER=true
OPENAI_QUERY_PLANNER_NON_ANSWER_VERIFY_CONFIDENCE=0.80
```

The defaults are shown above. The existing master switch remains:

```env
OPENAI_QUERY_PLANNER_ENABLED=true
```

## Installation

From the project root:

```powershell
Expand-Archive `
    "$env:USERPROFILE\Downloads\stage_7_7c_nonanswer_verifier_patch.zip" `
    -DestinationPath . `
    -Force
```

Back up the active planner:

```powershell
Copy-Item `
    .\app\legal_query_planner.py `
    .\app\legal_query_planner_before_7_7c.py `
    -Force
```

Inspect the diff:

```powershell
git diff --no-index -- `
    .\app\legal_query_planner.py `
    .\stage_7_7c_nonanswer_verifier\app\legal_query_planner.py
```

Install:

```powershell
Copy-Item `
    .\stage_7_7c_nonanswer_verifier\app\legal_query_planner.py `
    .\app\legal_query_planner.py `
    -Force

Copy-Item `
    .\stage_7_7c_nonanswer_verifier\scripts\test_stage_7_7c.py `
    .\scripts\test_stage_7_7c.py `
    -Force

Copy-Item `
    .\stage_7_7c_nonanswer_verifier\scripts\test_query_planner_stability.py `
    .\scripts\test_query_planner_stability.py `
    -Force
```

Compile:

```powershell
docker compose exec api `
    python -m py_compile `
    /app/app/legal_query_planner.py `
    /app/scripts/test_stage_7_7c.py `
    /app/scripts/test_query_planner_stability.py
```

Run offline checks:

```powershell
docker compose exec -T api `
    env PYTHONIOENCODING=utf-8 `
    python -m scripts.test_stage_7_7c
```

Restart API:

```powershell
docker compose restart api
```

Run the 13-case stability test three times:

```powershell
docker compose exec -T api `
    env PYTHONIOENCODING=utf-8 `
    python -m scripts.test_query_planner_stability `
    --repetitions 3 `
    --output /tmp/query_planner_stability_stage_7_7c.json
```

Copy the result:

```powershell
docker compose cp `
    api:/tmp/query_planner_stability_stage_7_7c.json `
    .\data\benchmarks\query_planner_stability_stage_7_7c.json
```

Do not rerun the 30-question regression benchmark until the stability test
reports all 13 cases stable and correct.
