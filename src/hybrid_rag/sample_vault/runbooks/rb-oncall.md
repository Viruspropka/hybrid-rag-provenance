---
id: rb-oncall
title: On-call rotation and escalation
type: runbook
status: active
tier: 1
reliability: 0.9
generated_by: human
verified: 2026-07-01
volatility: slow
tags: [oncall, paging, escalation]
---

# On-call rotation and escalation

## Rotation

The platform team runs a weekly on-call rotation. The handover happens every Monday at 10:00 UTC in a
short call where the outgoing engineer walks through open incidents and noisy alerts.

## Escalation

A page that is not acknowledged within 15 minutes escalates to the secondary on-call engineer. After
another 15 minutes it escalates to the engineering manager of the platform team. Severity 1 incidents
page the secondary at once.
