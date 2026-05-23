# Ring All-Reduce

`all_reduce` is one of the most important collectives in distributed training.
Data parallel training uses it to synchronize gradients across model replicas.

Ring all-reduce is a classic algorithm for implementing all-reduce efficiently
for large tensors.

## Problem

Assume four ranks. Each rank has a local tensor:

```text
rank 0: A
rank 1: B
rank 2: C
rank 3: D
```

After `all_reduce(sum)`, every rank should have:

```text
A + B + C + D
```

A naive algorithm could send everything to rank 0, reduce there, then broadcast
the result. That is correct, but rank 0 becomes a bottleneck.

Ring all-reduce avoids this central bottleneck.

## Ring Structure

Ranks are arranged in a logical ring:

```text
rank 0 -> rank 1 -> rank 2 -> rank 3 -> rank 0
```

Each rank sends to its next neighbor and receives from its previous neighbor.

The tensor is split into `world_size` chunks. With four ranks:

```text
tensor = [chunk 0, chunk 1, chunk 2, chunk 3]
```

## Two Phases

Ring all-reduce is commonly understood as two phases:

1. Reduce-scatter
2. All-gather

This is a crucial connection. `all_reduce` can be decomposed into:

```text
all_reduce = reduce_scatter + all_gather
```

## Phase 1: Reduce-Scatter

During reduce-scatter, chunks circulate around the ring. Each rank adds received
chunks to its local chunks.

After enough steps, every rank owns one fully reduced chunk.

For four ranks, the result after reduce-scatter is conceptually:

```text
rank 0 owns reduced chunk 0
rank 1 owns reduced chunk 1
rank 2 owns reduced chunk 2
rank 3 owns reduced chunk 3
```

Each reduced chunk already contains contributions from all ranks, but no rank
has the full tensor yet.

## Phase 2: All-Gather

During all-gather, the reduced chunks circulate again.

After enough steps, every rank receives every reduced chunk:

```text
rank 0 owns [reduced chunk 0, reduced chunk 1, reduced chunk 2, reduced chunk 3]
rank 1 owns [reduced chunk 0, reduced chunk 1, reduced chunk 2, reduced chunk 3]
rank 2 owns [reduced chunk 0, reduced chunk 1, reduced chunk 2, reduced chunk 3]
rank 3 owns [reduced chunk 0, reduced chunk 1, reduced chunk 2, reduced chunk 3]
```

Now each rank has the full all-reduce result.

## Why Ring Helps

Ring all-reduce has no central coordinator. Every rank sends and receives in each
step.

For large tensors, this is attractive because:

- Work is balanced across ranks.
- Each rank communicates with neighbors instead of funneling through one rank.
- Bandwidth can be used continuously.
- Memory can be handled chunk by chunk.

## Cost Intuition

With `p` ranks and `n` bytes per tensor, each rank sends and receives roughly:

```text
2 * (p - 1) / p * n bytes
```

Why `2`?

- One pass for reduce-scatter.
- One pass for all-gather.

Why `(p - 1) / p`?

- Each rank does not need to send its own final chunk to itself.

This is not the full real-world performance model, but it explains why ring
all-reduce is bandwidth efficient for large tensors.

## Relationship to Training Frameworks

| Context | Why ring all-reduce matters |
| --- | --- |
| PyTorch DDP | Gradients are commonly synchronized with all-reduce. |
| NCCL | NCCL may use ring-style algorithms depending on tensor size and topology. |
| Megatron-LM | Tensor-parallel layers often need collective communication inside model computation. |
| DeepSpeed | ZeRO and sharded optimizers make reduce-scatter/all-gather first-class operations. |

## Important Subtlety

The API may say `all_reduce`, but the implementation may internally behave like
reduce-scatter plus all-gather.

This matters because once we understand `reduce_scatter` and `all_gather`, ZeRO
and FSDP become much easier to reason about.

## What to Remember

- `all_reduce` gives every rank the same reduced tensor.
- Ring all-reduce avoids a single bottleneck rank.
- The algorithm splits tensors into chunks.
- It has two conceptual phases: reduce-scatter and all-gather.
- This decomposition shows up again in memory-efficient training frameworks.

