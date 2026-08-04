# Stage 8-B1 — Answer Prompt Hotfix

This is a prompt-only quality patch.

It changes only `GroundedAnswerGenerator._instructions()`.

It does not change:

- retrieval;
- routing;
- embeddings;
- KG traversal;
- article ranking;
- citation validation;
- generation schema;
- model or reasoning effort;
- `/retrieve`;
- retrieval benchmarks.

Expected style for the loan-deduction example:

```text
يجوز لصاحب العمل استرداد السلفة من أجر العامل، بشرط ألا يزيد كل قسط يُحسم
لاستردادها على 10% من الأجر [المادة 47].
```

For that simple question:

```json
{
  "key_points": [],
  "limitations": []
}
```
