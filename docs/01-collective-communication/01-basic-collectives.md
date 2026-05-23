# 基本 Collective

这篇笔记解释阅读 PyTorch、NCCL、Megatron-LM 和 DeepSpeed 源码之前必须掌握的第一组通信原语。

假设有 4 个 rank：

```text
rank 0 有 A
rank 1 有 B
rank 2 有 C
rank 3 有 D
```

## Broadcast

`broadcast(src=0)` 会把源 rank 的数据复制给所有 rank。

执行后：

```text
rank 0 有 A
rank 1 有 A
rank 2 有 A
rank 3 有 A
```

典型用途：

- 初始化时从 rank 0 同步模型参数。
- 分享配置或元数据。

关键点：

- 只有一个 rank 提供权威值。

## Reduce

`reduce(dst=0, op=sum)` 会把所有 rank 的值聚合起来，并把结果放到一个目标 rank 上。

执行后：

```text
rank 0 有 A + B + C + D
rank 1 可能仍保留 B，也可能输出未定义，取决于具体 API 语义
rank 2 可能仍保留 C，也可能输出未定义，取决于具体 API 语义
rank 3 可能仍保留 D，也可能输出未定义，取决于具体 API 语义
```

典型用途：

- 把指标聚合到一个 logging rank。
- 只需要一个 rank 拿到全局 scalar 的场景。

关键点：

- 所有人贡献数据，但只有一个 rank 得到最终值。

## All-Reduce

`all_reduce(op=sum)` 会把所有 rank 的值聚合起来，并把结果发给每个 rank。

执行后：

```text
rank 0 有 A + B + C + D
rank 1 有 A + B + C + D
rank 2 有 A + B + C + D
rank 3 有 A + B + C + D
```

典型用途：

- 数据并行中的梯度同步。

为什么重要：

- 在数据并行训练中，每个 rank 在不同数据 shard 上计算梯度。为了让模型副本保持一致，各 rank 在 optimizer step 前需要得到相同的平均梯度。

通常可以理解为：

```text
global_grad = sum(local_grad_i for each rank i) / world_size
```

PyTorch DDP 常见做法是执行 sum all-reduce，然后在梯度处理中除以 world size。

## All-Gather

`all_gather` 会从每个 rank 收集一个 tensor，并让每个 rank 都拿到完整列表。

执行后：

```text
rank 0 有 [A, B, C, D]
rank 1 有 [A, B, C, D]
rank 2 有 [A, B, C, D]
rank 3 有 [A, B, C, D]
```

典型用途：

- 重建一个被切分的 tensor。
- 收集被各 rank 持有的 activation 或 parameter shard。

关键点：

- 和 `all_reduce` 不同，`all_gather` 不做数学聚合，只是收集或拼接。

## Reduce-Scatter

`reduce_scatter(op=sum)` 会先对输入进行 reduce，然后把结果的不同 shard 分给不同 rank。

概念上：

```text
每个 rank 开始都有 chunks [x0, x1, x2, x3]
所有 rank 对对应 chunk 执行 reduce
rank 0 得到 reduced chunk 0
rank 1 得到 reduced chunk 1
rank 2 得到 reduced chunk 2
rank 3 得到 reduced chunk 3
```

典型用途：

- ZeRO 和 FSDP 风格设计中的梯度切分。
- 显存更高效的 distributed optimizer。

关键点：

- 它不会让每个 rank 都拿到完整 reduce 结果。
- 这种操作经常是显存节省背后的核心通信原语。

## All-To-All

`all_to_all` 允许每个 rank 向其他每个 rank 发送不同数据片段。

典型用途：

- Mixture-of-Experts 中的 token dispatch。
- Expert parallelism。
- 某些 sequence parallel 或 tensor 重分布模式。

关键点：

- 这是一个通用交换操作。它很强大，但也可能很昂贵，并且对负载均衡很敏感。

## Barrier

`barrier` 会让 group 中所有 rank 等到彼此都到达同一个位置。

典型用途：

- 调试。
- 确认文件或 checkpoint 已可见。
- 分隔性能测量阶段。

关键点：

- 它本身不移动模型数据。
- 过度使用 barrier 可能隐藏性能问题，也可能制造不必要的空等。

## 第一张训练映射表

| 训练机制 | 常见 collective |
| --- | --- |
| 初始参数同步 | `broadcast` |
| DDP 梯度同步 | `all_reduce` |
| sharded gradient reduce | `reduce_scatter` |
| sharded parameter materialization | `all_gather` |
| expert token routing | `all_to_all` |
| debug sync point | `barrier` |

下一步是理解通信代价。同一个数学结果可以用不同算法实现，而这些算法在 PCIe、NVLink、InfiniBand 或 Ethernet 上的表现会很不一样。
