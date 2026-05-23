# Basic Collectives

This note describes the first set of communication primitives we need before
reading PyTorch, NCCL, Megatron-LM, or DeepSpeed source code.

Assume four ranks:

```text
rank 0 has A
rank 1 has B
rank 2 has C
rank 3 has D
```

## Broadcast

`broadcast(src=0)` copies data from one source rank to all ranks.

After broadcast:

```text
rank 0 has A
rank 1 has A
rank 2 has A
rank 3 has A
```

Typical use:

- Synchronize model parameters from rank 0 at initialization.
- Share configuration or metadata.

Key point:

- Only one rank provides the authoritative value.

## Reduce

`reduce(dst=0, op=sum)` combines values from all ranks and stores the result on
one destination rank.

After reduce:

```text
rank 0 has A + B + C + D
rank 1 may keep B or have undefined output depending on API semantics
rank 2 may keep C or have undefined output depending on API semantics
rank 3 may keep D or have undefined output depending on API semantics
```

Typical use:

- Aggregate metrics to one logging rank.
- Collect a global scalar where only one rank needs the result.

Key point:

- Everyone contributes, but only one rank receives the final value.

## All-Reduce

`all_reduce(op=sum)` combines values from all ranks and gives the result to every
rank.

After all-reduce:

```text
rank 0 has A + B + C + D
rank 1 has A + B + C + D
rank 2 has A + B + C + D
rank 3 has A + B + C + D
```

Typical use:

- Data parallel gradient synchronization.

Why it matters:

- In data parallel training, each rank computes gradients on a different batch
  shard. To keep model replicas identical, ranks need the same averaged
  gradient before the optimizer step.

Usually:

```text
global_grad = sum(local_grad_i for each rank i) / world_size
```

PyTorch DDP often performs a sum all-reduce and applies division by world size
as part of gradient handling.

## All-Gather

`all_gather` collects one tensor from each rank and gives the full list to every
rank.

After all-gather:

```text
rank 0 has [A, B, C, D]
rank 1 has [A, B, C, D]
rank 2 has [A, B, C, D]
rank 3 has [A, B, C, D]
```

Typical use:

- Reconstruct a sharded tensor.
- Gather activations or parameters that were partitioned across ranks.

Key point:

- Unlike all-reduce, values are not mathematically combined. They are
  concatenated or collected.

## Reduce-Scatter

`reduce_scatter(op=sum)` first reduces values, then scatters different chunks of
the reduced result to different ranks.

Conceptually:

```text
each rank starts with chunks [x0, x1, x2, x3]
all ranks reduce corresponding chunks
rank 0 receives reduced chunk 0
rank 1 receives reduced chunk 1
rank 2 receives reduced chunk 2
rank 3 receives reduced chunk 3
```

Typical use:

- Gradient sharding in ZeRO and FSDP-like designs.
- Memory-efficient distributed optimizer implementations.

Key point:

- It avoids giving the full reduced tensor to every rank.
- This is often the communication primitive behind memory savings.

## All-To-All

`all_to_all` lets every rank send a different piece of data to every other rank.

Typical use:

- Mixture-of-Experts token dispatch.
- Expert parallelism.
- Some sequence-parallel or tensor-redistribution patterns.

Key point:

- This is a general exchange. It is powerful but can be expensive and sensitive
  to load imbalance.

## Barrier

`barrier` forces all ranks in a group to wait until every rank reaches the same
point.

Typical use:

- Debugging.
- Ensuring files or checkpoints are visible before continuing.
- Separating measurement phases.

Key point:

- It does not move model data by itself.
- Overusing barriers can hide performance problems or create unnecessary idle
  time.

## First Training Mapping

| Training mechanism | Common collective |
| --- | --- |
| Initial parameter sync | `broadcast` |
| DDP gradient sync | `all_reduce` |
| Sharded gradient reduce | `reduce_scatter` |
| Sharded parameter materialization | `all_gather` |
| Expert token routing | `all_to_all` |
| Debug sync point | `barrier` |

The next step is to understand communication cost. The same mathematical result
can be implemented by different algorithms, and those algorithms behave very
differently on PCIe, NVLink, InfiniBand, or Ethernet.

