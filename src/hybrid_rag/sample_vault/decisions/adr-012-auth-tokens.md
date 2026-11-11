---
id: adr-012-auth-tokens
title: "ADR-012: Short-lived access tokens with refresh rotation"
type: decision
status: active
tier: 1
reliability: 0.95
generated_by: human
verified: 2026-04-03
volatility: slow
tags: [adr, auth, security]
---

# ADR-012: Short-lived access tokens with refresh rotation

## Decision

API access tokens expire after 15 minutes. Clients get a new access token with a refresh token, which is
valid for 30 days and is rotated on every use; a refresh token that is used twice revokes the whole session.

## Why

Long-lived tokens leaked into logs twice in one year. Short lifetimes limit the damage of a leak, and
rotation makes a stolen refresh token visible.
