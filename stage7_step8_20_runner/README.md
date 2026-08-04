# Stage 7 → Step 8 runner

The script reads the 20 questions, calls `/retrieve`, and sends the exact Step 7 JSON response directly to `/generate`.

```bash
pip install requests

python scripts/run_stage7_step8_20.py \
  --questions data/retrieval_benchmark_unseen_20_stage_7_7d.json \
  --output-dir data/manual_review/stage7_step8_20
```

Add `--retrieve-debug` or `--generate-debug` only when needed.
