# Roadmap

This roadmap follows the execution path of distributed training instead of the
marketing boundary of each framework.

## Phase 1: Collective Communication

Goal: understand the communication operations that every later framework builds
on.

Topics:

- Rank, world size, process group, backend
- Point-to-point vs collective communication
- `broadcast`, `reduce`, `all_reduce`, `all_gather`, `reduce_scatter`,
  `scatter`, `gather`, `all_to_all`, `barrier`
- Latency, bandwidth, topology, synchronization
- Ring, tree, hierarchical, and topology-aware algorithms
- Why gradient synchronization often maps to `all_reduce`
- Why tensor parallelism often uses `all_reduce`, `all_gather`, and
  `reduce_scatter`

Output:

- Primitive-by-primitive notes
- Small PyTorch distributed examples
- Communication pattern diagrams

## Phase 2: PyTorch Distributed

Goal: connect the API we write to the backend behavior that actually runs.

Topics:

- `torch.distributed.init_process_group`
- Stores, rendezvous, rank assignment, launchers
- Process groups and subgroups
- `ProcessGroupNCCL`
- DDP gradient buckets and overlap
- FSDP and reduce-scatter/all-gather style sharding

Output:

- Minimal runnable examples
- API-to-source call chains
- DDP/FSDP communication comparison

## Phase 3: NCCL

Goal: understand NCCL as the GPU communication engine.

Topics:

- Communicators, ranks, channels, streams
- Rings, trees, protocols, and topology
- `ncclAllReduce`, `ncclBroadcast`, `ncclReduceScatter`, `ncclAllGather`
- Intra-node NVLink/PCIe and inter-node networking
- Debugging with `NCCL_DEBUG`, `NCCL_TOPO_DUMP_FILE`, and environment variables

Output:

- NCCL concept map
- Source reading notes for selected functions
- Debugging checklist

## Phase 4: Megatron-LM

Goal: understand how parallel training strategies compose.

Topics:

- Data parallelism, tensor parallelism, pipeline parallelism
- Model-parallel process group construction
- Column-parallel and row-parallel linear layers
- Sequence parallelism
- Pipeline schedules and microbatches
- Distributed optimizer behavior

Output:

- Process group diagrams
- Layer-level communication notes
- One training step walkthrough

## Phase 5: DeepSpeed

Goal: understand DeepSpeed as a training engine and memory optimizer.

Topics:

- Engine initialization and training loop wrapper
- ZeRO stages 1, 2, and 3
- Parameter, gradient, and optimizer-state partitioning
- Communication hooks and overlap
- Checkpointing and offload
- Integration with Megatron-style training

Output:

- ZeRO communication/memory tables
- DeepSpeed engine call chain
- Comparison with PyTorch DDP/FSDP and Megatron distributed optimizer

## Phase 6: End-to-End Training

Goal: explain one complete distributed training iteration.

Topics:

- Launch
- Process group initialization
- Model construction
- Parallel wrapping
- Data loading
- Forward pass
- Loss computation
- Backward pass
- Gradient communication
- Optimizer step
- Checkpointing and logging

Output:

- Full training-step diagram
- A minimal PyTorch distributed training demo
- A comparison of PyTorch DDP, Megatron-LM, and DeepSpeed flows

## Study Method

Each topic should be documented with the same pattern:

1. Concept: what problem does it solve?
2. API: what code does the user write?
3. Runtime: what happens across ranks?
4. Source: where is the implementation?
5. Experiment: how can we verify it?
6. Summary: what should we remember?

