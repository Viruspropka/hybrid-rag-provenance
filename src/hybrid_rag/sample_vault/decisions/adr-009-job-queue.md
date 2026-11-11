---
id: adr-009-job-queue
title: "ADR-009: Move the job queue to Redis Streams"
type: decision
status: active
tier: 1
reliability: 0.95
generated_by: human
verified: 2026-02-11
volatility: slow
tags: [adr, queue, architecture, redis]
---

# ADR-009: Move the job queue to Redis Streams

## Decision

The job queue moves from the SQLite table of ADR-003 to Redis Streams with one consumer group per job type.
Workers acknowledge a job only after it has finished, so a crashed worker's jobs are claimed again after
five minutes.

## Why

Job volume grew twenty times in a year and a single worker host became the bottleneck. Redis is already
operated for sessions, so the team adds no new technology.

## Consequences

Consumer group lag becomes the main health signal of the queue and gets its own alert.
