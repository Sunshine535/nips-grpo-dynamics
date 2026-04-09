---
type: paper
node_id: paper:rlvr-limit2025
title: "Does RL Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?"
authors: ["RLVR Limit Authors"]
year: 2025
venue: NeurIPS 2025 BPR
external_ids:
  arxiv: "2504.13837"
tags: [rlvr, capacity-bound, reasoning]
relevance: core
---

# Shows RLVR does not expand reasoning capacity beyond the base model but rather compresses existing search capabilities into more efficient generation.

## Relevance to This Project
CSD provides the theoretical explanation for this capacity bound: self-distillation cannot exceed the teacher's capacity, and since the teacher IS the online policy, RLVR is inherently bounded by base model capabilities.
