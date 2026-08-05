# Jordanian Labor Law KG GraphRAG

This project implements an ontology-driven Knowledge Graph and GraphRAG pipeline for the Jordanian Labor Law.

## Main Pipeline

```text
User Question
    ↓
Question Analysis
    ↓
LLM Query Planner
    ↓
Hybrid Retrieval
    ↓
Knowledge Graph Expansion
    ↓
Legal Article Reranking
    ↓
Grounded Answer Generation
    ↓
Answer with Legal Citations