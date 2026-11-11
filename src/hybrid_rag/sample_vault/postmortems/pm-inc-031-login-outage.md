---
id: pm-inc-031-login-outage
title: "INC-031 postmortem: login outage"
type: postmortem
status: archived
tier: 1
reliability: 0.9
generated_by: human
verified: 2025-12-05
volatility: slow
tags: [incident, postmortem, auth]
---

# INC-031 postmortem: login outage

## What happened

For 47 minutes nobody could log in to Tidewell. The token signing key had expired and the service kept
issuing tokens that every API node rejected.

## Root cause and actions

Key rotation was a manual yearly task that nobody owned. Rotation is now automated monthly and an alert
fires 14 days before any signing key expires.
