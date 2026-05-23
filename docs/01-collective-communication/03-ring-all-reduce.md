# Ring All-Reduce

`all_reduce` 是分布式训练中最重要的 collective 之一。数据并行训练用它在模型副本之间同步梯度。

Ring all-reduce 是一种经典算法，常用于高效实现大 tensor 的 all-reduce。

## 问题

假设有 4 个 rank，每个 rank 有一个本地 tensor：

```text
rank 0: A
rank 1: B
rank 2: C
rank 3: D
```

执行 `all_reduce(sum)` 后，每个 rank 都应该得到：

```text
A + B + C + D
```

一种朴素算法是：所有 rank 把数据发给 rank 0，在 rank 0 上 reduce，再把结果 broadcast 回去。这个算法是正确的，但 rank 0 会成为瓶颈。

Ring all-reduce 的目标就是避免这个中心瓶颈。

## Ring 结构

把 rank 组织成一个逻辑环：

```text
rank 0 -> rank 1 -> rank 2 -> rank 3 -> rank 0
```

每个 rank 向下一个邻居发送数据，并从上一个邻居接收数据。

tensor 会被切成 `world_size` 个 chunk。4 个 rank 时：

```text
tensor = [chunk 0, chunk 1, chunk 2, chunk 3]
```

## 两个阶段

Ring all-reduce 通常可以理解成两个阶段：

1. Reduce-scatter
2. All-gather

这是一个非常关键的连接：

```text
all_reduce = reduce_scatter + all_gather
```

## 阶段 1：Reduce-Scatter

在 reduce-scatter 阶段，chunk 会沿着 ring 流动。每个 rank 把收到的 chunk 加到自己的本地 chunk 上。

经过足够多步后，每个 rank 会拥有一个已经完全 reduce 好的 chunk。

对于 4 个 rank，reduce-scatter 结束后概念上是：

```text
rank 0 拥有 reduced chunk 0
rank 1 拥有 reduced chunk 1
rank 2 拥有 reduced chunk 2
rank 3 拥有 reduced chunk 3
```

每个 reduced chunk 已经包含所有 rank 的贡献，但还没有任何一个 rank 拥有完整 tensor。

## 阶段 2：All-Gather

在 all-gather 阶段，reduced chunk 再次沿着 ring 流动。

经过足够多步后，每个 rank 都拿到所有 reduced chunk：

```text
rank 0 拥有 [reduced chunk 0, reduced chunk 1, reduced chunk 2, reduced chunk 3]
rank 1 拥有 [reduced chunk 0, reduced chunk 1, reduced chunk 2, reduced chunk 3]
rank 2 拥有 [reduced chunk 0, reduced chunk 1, reduced chunk 2, reduced chunk 3]
rank 3 拥有 [reduced chunk 0, reduced chunk 1, reduced chunk 2, reduced chunk 3]
```

此时每个 rank 都得到了完整 all-reduce 结果。

## 为什么 Ring 有用

Ring all-reduce 没有中心协调者。每个 rank 在每一步都发送和接收。

对于大 tensor，这很有吸引力：

- 各 rank 工作量更均衡。
- 不会把所有数据 funnel 到一个 rank。
- bandwidth 可以被持续利用。
- 可以按 chunk 处理内存。

## 代价直觉

有 `p` 个 rank，每个 tensor 有 `n` 字节时，每个 rank 大约发送和接收：

```text
2 * (p - 1) / p * n bytes
```

为什么有 `2`？

- 一轮 reduce-scatter。
- 一轮 all-gather。

为什么有 `(p - 1) / p`？

- 每个 rank 不需要把自己的最终 chunk 发给自己。

这不是完整真实世界性能模型，但它解释了为什么 ring all-reduce 对大 tensor 具有较好的带宽效率。

## 与训练框架的关系

| 场景 | 为什么 ring all-reduce 重要 |
| --- | --- |
| PyTorch DDP | 梯度通常通过 all-reduce 同步。 |
| NCCL | NCCL 可能根据 tensor 大小和 topology 选择 ring 风格算法。 |
| Megatron-LM | 张量并行 layer 常在模型计算内部使用 collective。 |
| DeepSpeed | ZeRO 和 sharded optimizer 让 reduce-scatter/all-gather 成为核心操作。 |

## 一个重要细节

API 可能叫 `all_reduce`，但实现内部可能表现为 reduce-scatter 加 all-gather。

这很重要，因为理解 `reduce_scatter` 和 `all_gather` 之后，ZeRO 与 FSDP 会更容易理解。

## 应该记住什么

- `all_reduce` 会让每个 rank 得到相同的 reduce 后 tensor。
- Ring all-reduce 避免单个瓶颈 rank。
- 算法会把 tensor 切成 chunk。
- 它可以分解为两个概念阶段：reduce-scatter 和 all-gather。
- 这种分解会在显存高效训练框架中反复出现。
