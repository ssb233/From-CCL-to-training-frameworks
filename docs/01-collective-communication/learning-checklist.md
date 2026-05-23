# Learning Checklist

Use this checklist to track the collective communication chapter.

## Concepts

- [x] Understand rank, world size, process group, and backend.
- [x] Distinguish point-to-point communication from collectives.
- [x] Explain `broadcast`.
- [x] Explain `reduce`.
- [x] Explain `all_reduce`.
- [x] Explain `all_gather`.
- [x] Explain `reduce_scatter`.
- [x] Explain `all_to_all`.
- [x] Explain `barrier`.
- [x] Explain latency and bandwidth.
- [x] Explain why bucketing helps DDP.
- [x] Explain ring all-reduce at a high level.
- [ ] Explain tree collectives.
- [ ] Explain hierarchical collectives.

## Training Mapping

- [x] Map DDP gradient sync to `all_reduce`.
- [x] Map ZeRO/FSDP gradient sharding to `reduce_scatter`.
- [x] Map ZeRO/FSDP parameter materialization to `all_gather`.
- [x] Map expert routing to `all_to_all`.
- [ ] Map Megatron tensor-parallel linear layers to concrete collectives.

## Experiments

- [ ] Run a local `all_reduce` example.
- [ ] Run a local `all_gather` example.
- [ ] Run a local `reduce_scatter` example.
- [ ] Inspect how tensor values change per rank.
- [ ] Add diagrams for the core collectives.

