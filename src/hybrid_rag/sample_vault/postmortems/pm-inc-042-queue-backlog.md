---
id: pm-inc-042-queue-backlog
title: "INC-042 postmortem: job queue backlog"
type: postmortem
status: active
tier: 1
reliability: 0.9
generated_by: human
verified: 2026-10-20
volatility: slow
tags: [incident, postmortem, queue]
---

# INC-042 postmortem: job queue backlog

## What happened

On 2026-10-18 exports and email digests were delayed for three hours. The job queue backlog grew to
40 thousand jobs while the workers looked healthy.

## Root cause

After a planned Redis failover the consumer group of the export workers pointed at a stale stream id, so the
workers read nothing. The consumer group lag alert existed but had a threshold of one million jobs.

## Actions

Lower the lag alert threshold to 5 thousand jobs, add a failover check to the Redis runbook, and add a
synthetic job that must complete every five minutes.
