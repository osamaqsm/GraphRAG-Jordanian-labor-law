# Stage 7.7-A strict-schema hotfix

## Cause

`LegalQueryPlan.model_json_schema()` omits Pydantic fields with defaults from
an object's `required` array. OpenAI strict Structured Outputs requires every
property at every object level to be present in `required` and every object to
set `additionalProperties: false`.

The first rejected nested property was `actors`. Other fields would have had
the same problem.

## Fix

`LegalQueryPlanner._strict_response_schema()` now:

1. Builds the Pydantic JSON Schema.
2. Recursively sets `required` to every key in `properties`.
3. Recursively sets `additionalProperties` to `false`.
4. Removes Pydantic `default` keywords from the API schema.

Application semantics remain unchanged. Fields that are semantically optional
are returned as empty strings or empty lists.

## Files

- `app/legal_query_planner.py` — application hotfix.
- `scripts/test_stage_7_7a.py` — adds recursive strict-schema checks.

## Install

```powershell
Copy-Item `
    .\app\legal_query_planner.py `
    .\app\legal_query_planner_before_schema_fix.py `
    -Force

Copy-Item `
    .\stage_7_7a_schema_fix\app\legal_query_planner.py `
    .\app\legal_query_planner.py `
    -Force

Copy-Item `
    .\stage_7_7a_schema_fix\scripts\test_stage_7_7a.py `
    .\scripts\test_stage_7_7a.py `
    -Force
```

## Validate

```powershell
docker compose exec api `
    python -m py_compile `
    /app/app/legal_query_planner.py `
    /app/scripts/test_stage_7_7a.py `
    /app/scripts/test_query_planner_live.py
```

```powershell
docker compose exec -T api `
    env PYTHONIOENCODING=utf-8 `
    python -m scripts.test_stage_7_7a
```

Expected additional line:

```text
Strict OpenAI schema: every object property is required
```

Then recreate the API container and rerun the live test:

```powershell
docker compose up -d --force-recreate api

docker compose exec -T api `
    env PYTHONIOENCODING=utf-8 `
    python -m scripts.test_query_planner_live
```

The live test should now print structured planner results rather than the
`Missing 'actors'` schema error.
