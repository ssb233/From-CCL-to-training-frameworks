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
2. Communication cost model
3. Ring all-reduce
4. Tree and hierarchical collectives
5. Mapping collectives to distributed training
6. Minimal PyTorch examples

