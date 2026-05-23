# Distributed Training Study Tracker

Last updated: 2026-05-23

This file is the single source of truth for long-term progress.

## Chapter Progress

| Chapter | Status | Notes |
| --- | --- | --- |
| 01 Collective Communication | In progress | Basic collectives, cost model, ring all-reduce drafted |
| 02 PyTorch Distributed | Not started | |
| 03 NCCL | Not started | |
| 04 Megatron-LM | Not started | |
| 05 DeepSpeed | Not started | |
| 06 End-to-End Training Loop | Not started | |

## Topics Mastered

| Topic | Date | Confidence | Evidence |
| --- | --- | --- | --- |
| Repository learning structure | 2026-05-23 | Medium-High | Repository initialized with roadmap and docs |

## Active Knowledge Gaps

| Gap | Severity | Related chapter | Follow-up |
| --- | --- | --- | --- |
| Need hands-on PyTorch collective experiments | Medium | 01 Collective Communication | Create and run `all_reduce_demo.py` |
| Need source-level path from PyTorch API to NCCL backend | High | 02 PyTorch Distributed | Clone or inspect pinned PyTorch source |

## Source Reading Status

| Project | Status | Version or commit | Notes |
| --- | --- | --- | --- |
| PyTorch | Not started | | |
| NCCL | Not started | | |
| Megatron-LM | Not started | | |
| DeepSpeed | Not started | | |

## Experiment Status

| Experiment | Status | Notes |
| --- | --- | --- |
| `all_reduce_demo.py` | Not started | First priority |
| `all_gather_demo.py` | Not started | |
| `reduce_scatter_demo.py` | Not started | |
| DDP gradient sync demo | Not started | |

## Next Recommended Topics

1. Run a local `all_reduce` experiment.
2. Add diagrams for basic collectives.
3. Study tree and hierarchical collectives.
4. Start PyTorch distributed API path: `dist.all_reduce`.

