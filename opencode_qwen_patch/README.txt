OpenCode Go / Qwen3.7 Plus patch

Replace:
  app/config.py
  app/llm_provider.py

Add to .env:
  OPENCODE_API_KEY=<your key>
  OPENCODE_BASE_URL=https://opencode.ai/zen/go
  OPENCODE_TIMEOUT_SECONDS=120

For Qwen smoke/final runs:
  PIPELINE_LLM_PROVIDER=opencode
  PIPELINE_LLM_MODEL=qwen3.7-plus

Do not change any other benchmark/pipeline settings.
