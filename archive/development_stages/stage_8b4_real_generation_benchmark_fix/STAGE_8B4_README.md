# Stage 8-B4 — Real-Retrieval Generation Benchmark Fix

The first real-retrieval run is structurally valid, but it exposed two
measurement problems:

1. The old generator rejected six answers because citation formatting did not
   exactly match the structured citation list.
2. The benchmark matcher treated Arabic digits, number words, morphology, and
   word order as different legal facts.

This patch:

- installs the citation-normalizing generator;
- preserves rejected drafts in debug output;
- strengthens prompt fidelity for legal conditions and numbers;
- creates benchmark rubric v1.1 without changing any frozen retrieval file;
- keeps genuine failures strict.

Do not recollect retrieval. Reuse the same 20 frozen retrieval JSON files.
