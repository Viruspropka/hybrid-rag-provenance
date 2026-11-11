---
id: ref-api-rate-limits
title: Public API rate limits
type: reference
status: active
tier: 0
reliability: 0.95
generated_by: human
verified: 2026-10-01
volatility: slow
tags: [api, limits, reference]
---

# Public API rate limits

## Limits per token

Each API token may send 600 requests per minute, with a burst allowance of 50 requests. Export
endpoints have a separate limit of 10 requests per minute per token.

## What clients see

A client over the limit gets HTTP 429 with a Retry-After header in seconds. Limits are counted per
token, not per IP address, so several services sharing one token share one budget.
