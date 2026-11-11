---
id: adr-003-job-queue
title: "ADR-003: SQLite-backed job queue"
type: decision
status: superseded
tier: 1
reliability: 0.9
generated_by: human
verified: 2024-05-02
volatility: slow
tags: [adr, queue, architecture]
---

# ADR-003: SQLite-backed job queue

## Decision

Background jobs (thumbnail rendering, email digests, exports) run from a job queue stored in a SQLite table
on the worker host. One worker process polls the table every second.

## Why

The job volume is small, the team has no operations capacity for another service, and SQLite is already
deployed. Superseded by ADR-009 once the volume outgrew a single worker host.
