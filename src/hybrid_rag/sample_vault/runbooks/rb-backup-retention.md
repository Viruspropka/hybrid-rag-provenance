---
id: rb-backup-retention
title: Backup retention runbook
type: runbook
status: active
tier: 1
reliability: 0.95
generated_by: human
verified: 2026-09-14
volatility: slow
tags: [backups, storage, runbook]
---

# Backup retention runbook

## Retention period

Nightly database backups of the Tidewell primary cluster start at 02:00 UTC and are kept for 90 days.
Monthly snapshots taken on the first day of the month are kept for one year in cold storage. The
lifecycle rule that deletes older backups is reviewed every six months by the platform team.

## Restore drill

A restore drill runs every month on the first Tuesday. The on-call engineer restores the latest nightly
backup into the staging cluster, runs the integrity checks and records the restore time. The target
recovery time objective is two hours and the recovery point objective is 24 hours.

## When a backup fails

The backup job pages the on-call engineer after two consecutive failures. Check free space on the backup
bucket first, then the credentials of the backup service account.
