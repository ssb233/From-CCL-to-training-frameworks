# Collective Communication

Collective communication is the foundation of distributed training. It describes
operations where a group of ranks coordinate data movement together.

The first objective is not to memorize APIs. The first objective is to build an
intuition for what each operation does to data across ranks.

## Core Vocabulary

- Rank: one participant in a distributed job.
- World size: total number of ranks in a process group.
- Process group: a set of ranks that can communicate.
- Backend: the implementation that executes communication, such as NCCL or Gloo.
- Tensor: the data unit usually communicated by PyTorch distributed APIs.
- Collective: an operation called by every rank in the group.

## Mental Model

Most collectives answer one of four questions:

- Should one rank send data to everyone?
- Should everyone contribute data into one result?
- Should everyone receive the same combined result?
- Should everyone exchange different pieces of data?

This gives us the first map:

| Operation | Main idea | Common training use |
| --- | --- | --- |
| `broadcast` | One rank sends the same tensor to all ranks | Sync initial parameters |
| `reduce` | All ranks contribute, one rank receives result | Metrics or centralized aggregation |
| `all_reduce` | All ranks contribute, all ranks receive result | Gradient averaging in DDP |
| `all_gather` | All ranks share local tensors with all ranks | Reconstruct sharded tensors |
| `reduce_scatter` | Reduce all inputs, then scatter shards | ZeRO/FSDP-style gradient sharding |
| `all_to_all` | Every rank sends distinct chunks to every other rank | Expert parallelism, token dispatch |
| `barrier` | All ranks wait until everyone arrives | Debugging and synchronization |

## Learning Order

1. [Basic collectives](01-basic-collectives.md)
2. [Communication cost model](02-communication-cost-model.md)
3. [Ring all-reduce](03-ring-all-reduce.md)
4. Tree and hierarchical collectives
5. Mapping collectives to distributed training
6. Minimal PyTorch examples

## First Reading Questions

Before touching NCCL or PyTorch source code, we want to answer these questions:

- If every rank has a gradient tensor, why does DDP need `all_reduce`?
- Why is `all_reduce` often implemented as `reduce_scatter` plus `all_gather`?
- Why does communication time depend on both latency and bandwidth?
- Why does topology matter even when the API call is the same?
- Why can two correct collective algorithms have very different performance?

## Chapter Outputs

By the end of this chapter, we should have:

- A vocabulary for reading distributed training code.
- A table mapping training behavior to collective primitives.
- A cost model for reasoning about performance.
- A worked explanation of ring all-reduce.
- Minimal PyTorch examples for each important primitive.
