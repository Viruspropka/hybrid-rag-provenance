---
id: rb-deploy
title: Deploy and rollback runbook
type: runbook
status: active
tier: 1
reliability: 0.95
generated_by: human
verified: 2026-08-20
volatility: slow
tags: [deploy, release, rollback]
---

# Deploy and rollback runbook

## Normal deploy

Tidewell ships with blue-green deploys. The new version starts on the idle colour, receives a canary
share of 5 percent of traffic for 30 minutes, and is promoted to 100 percent when the error rate and the
p95 latency stay inside the release budget.

## Rolling back a bad release

If the canary breaks the release budget the pipeline rolls back on its own. To roll back by hand run
`tidewell-deploy rollback --to previous`. Traffic switches back to the previous colour within a minute;
database migrations are never rolled back automatically, so check the migration notes first.

## Freeze windows

No deploys on Friday after 15:00 UTC and during the last two days of a month, when invoices are generated.
