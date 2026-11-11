---
id: adr-014-vector-store
title: "ADR-014: Vector search inside the existing Postgres (pgvector)"
type: decision
status: active
tier: 1
reliability: 0.95
generated_by: human
verified: 2026-06-30
volatility: slow
tags: [adr, search, vector, database]
---

# ADR-014: Vector search inside the existing Postgres (pgvector)

## Decision

Semantic search for help-centre articles uses the pgvector extension in the existing Postgres cluster.
We do not run a separate vector database.

## Why

The corpus is below one million vectors, backups and access control already exist for Postgres, and a
benchmark on our own articles showed no recall gain from a dedicated vector database at this size.
The research note from 2025 recommended a dedicated service; this decision replaces that recommendation.
