---
type: paper
node_id: paper:clipo2026
title: "CLIPO: Contrastive Learning in Policy Optimization Generalizes RLVR"
authors: ["CLIPO Authors"]
year: 2026
venue: arXiv
external_ids:
  arxiv: "2603.10101"
tags: [grpo, contrastive-learning, InfoNCE]
relevance: core
---

# Adds an InfoNCE-style contrastive head to GRPO, framing policy optimization as contrastive learning over response groups.

## Relevance to This Project
CSD proves GRPO already IS contrastive (group normalization induces implicit contrastive structure), making CLIPO's explicit contrastive head redundant architectural overhead.
