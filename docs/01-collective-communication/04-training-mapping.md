# Mapping Collectives to Training

Collectives are not abstract math exercises. Each one appears because a training
strategy creates a specific data movement problem.

## Data Parallel Training

In data parallelism:

- Every rank has a full model replica.
- Every rank processes different input data.
- Every rank computes local gradients.
- Model replicas must stay numerically synchronized.

The core collective is:

```text
all_reduce(local_gradients)
```

After all-reduce and averaging, every rank applies the same optimizer update.

## Tensor Parallel Training

In tensor parallelism:

- A layer is split across ranks.
- Each rank computes part of the layer output or gradient.
- Some layer boundaries require combining partial results.

Common collectives:

- `all_reduce`
- `all_gather`
- `reduce_scatter`

Example intuition:

- If ranks compute partial sums, use `all_reduce`.
- If ranks own shards that must be materialized together, use `all_gather`.
- If ranks compute a full result but should keep only shards, use
  `reduce_scatter`.

## Pipeline Parallel Training

In pipeline parallelism:

- Different ranks own different layers.
- Activations move forward between stages.
- Gradients move backward between stages.

This often uses point-to-point communication rather than only collectives.

Important idea:

```text
not every distributed training communication is a collective
```

Pipeline parallelism forces us to study send/recv later.

## ZeRO and Sharded Training

In ZeRO-style training:

- Optimizer states may be partitioned.
- Gradients may be partitioned.
- Parameters may be partitioned.

Common collectives:

- `reduce_scatter` to reduce gradients and keep only shards.
- `all_gather` to materialize parameters when needed.

The tradeoff:

```text
less memory replication, more structured communication
```

## Mixture-of-Experts

In expert parallelism:

- Tokens are routed to different expert ranks.
- Each rank may send different token subsets to other ranks.

The common collective is:

```text
all_to_all
```

This communication pattern is harder to balance because token routing may be
uneven.

## Summary Table

| Training pattern | Data movement problem | Common communication |
| --- | --- | --- |
| DDP | Average gradients across replicas | `all_reduce` |
| FSDP / ZeRO | Reduce and shard gradients | `reduce_scatter` |
| FSDP / ZeRO | Materialize sharded parameters | `all_gather` |
| Tensor parallel linear layer | Combine partial results | `all_reduce` or `all_gather` |
| Pipeline parallelism | Move activations and gradients between stages | send/recv |
| Expert parallelism | Route tokens to experts | `all_to_all` |

## Reading Direction

When we read framework source code, we should always ask:

1. What tensor exists on this rank right now?
2. Which ranks need this tensor next?
3. Should values be copied, reduced, scattered, or exchanged?
4. Is the communication on the critical path, or can it overlap with compute?

