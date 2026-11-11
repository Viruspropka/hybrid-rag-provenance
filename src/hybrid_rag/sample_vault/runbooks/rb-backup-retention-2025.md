---
id: rb-backup-retention-2025
title: Backup retention runbook (2025 edition)
type: runbook
status: superseded
tier: 1
reliability: 0.9
generated_by: human
verified: 2025-03-10
volatility: slow
tags: [backups, storage, runbook]
---

# Backup retention runbook (2025 edition)

## Retention period

Nightly database backups of the Tidewell primary cluster are kept for 30 days. After 30 days the
storage lifecycle rule deletes them automatically. Weekly snapshots are not kept separately.

## Restore drill

A restore drill runs once per quarter. The on-call engineer restores the latest nightly backup into
the staging cluster and checks that the row counts of the five largest tables match production.
