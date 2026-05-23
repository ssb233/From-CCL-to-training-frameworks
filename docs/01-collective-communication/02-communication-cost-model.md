# Communication Cost Model

Collective communication is not only about correctness. The same mathematical
operation can be correct but slow if it moves data in the wrong pattern.

This note builds the first performance model we need before reading NCCL,
PyTorch DDP, Megatron-LM, or DeepSpeed.

## The Two Basic Costs

A simple communication model separates time into two parts:

```text
communication_time ~= latency_cost + bandwidth_cost
```

Latency is the startup cost of sending a message. Bandwidth is the cost of
moving the bytes.

A common form is:

```text
time ~= alpha + n / beta
```

Where:

- `alpha` is latency.
- `n` is the number of bytes.
- `beta` is bandwidth in bytes per second.

This model is simple, but it is already enough to explain many distributed
training behaviors.

## Small Messages vs Large Messages

For small tensors, latency dominates.

```text
small n:
time ~= alpha
```

For large tensors, bandwidth dominates.

```text
large n:
time ~= n / beta
```

This is why frameworks often combine many small gradients into larger buckets.
One large all-reduce is usually better than thousands of tiny all-reduces.

## Number of Ranks Matters

Adding ranks can reduce compute per rank, but it often increases communication
coordination.

More ranks can mean:

- More participants that must enter a collective.
- More network links involved.
- More chances for one slow rank to delay everyone.
- More complex topology choices.

This is the first reason distributed training does not scale perfectly.

## Topology Matters

The API call hides topology:

```python
dist.all_reduce(tensor)
```

But the runtime has to choose a real path through hardware.

Within one node, common links include:

- NVLink
- PCIe
- GPU-to-CPU memory paths

Across nodes, common links include:

- InfiniBand
- Ethernet
- RoCE

The same `all_reduce` can have very different performance depending on whether
data moves over NVLink, PCIe, or inter-node network links.

## Synchronization Cost

A collective is called by every rank in the process group.

If one rank arrives late, other ranks may wait.

Common causes:

- Load imbalance in data loading.
- Uneven compute time.
- Straggler GPU or node.
- Different tensor shapes or conditional execution bugs.
- Network congestion.

This is why communication performance is also a systems problem, not only an
algorithm problem.

## Bucketing

Distributed training frameworks often group tensors into buckets before
communication.

Why:

- Fewer collective calls reduce latency overhead.
- Larger messages use bandwidth more efficiently.
- Buckets can overlap communication with backward computation.

In PyTorch DDP, gradients are organized into buckets. As backward computation
produces gradients, DDP can launch all-reduce for ready buckets before the whole
backward pass finishes.

The important idea:

```text
communication does not have to wait until all computation is finished
```

Good frameworks try to overlap communication and computation.

## A First Scaling Intuition

Suppose each rank computes gradients for the same model replica.

More ranks:

- Reduce data processed per rank if global batch size is fixed.
- Increase total batch size if per-rank batch size is fixed.
- Require gradient synchronization across more ranks.

The training step time becomes roughly:

```text
step_time ~= compute_time + exposed_communication_time
```

The word `exposed` matters. Some communication can be hidden behind backward
computation. Some cannot.

## What This Means for Frameworks

| Framework area | Cost-model question |
| --- | --- |
| PyTorch DDP | How are gradient buckets formed and reduced? |
| NCCL | Which algorithm and topology path does the backend choose? |
| Megatron-LM | Which layer operations force communication inside forward/backward? |
| DeepSpeed ZeRO | Which tensors are partitioned to reduce memory, and what communication replaces replication? |

## Key Takeaways

- Collective communication has both latency and bandwidth costs.
- Small messages are latency-sensitive.
- Large messages are bandwidth-sensitive.
- Bucketing improves efficiency and enables overlap.
- Topology changes the real cost behind the same API.
- Distributed training performance depends on how much communication remains
  exposed after overlap.

