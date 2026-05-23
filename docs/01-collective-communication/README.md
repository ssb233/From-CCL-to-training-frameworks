# 集合通信

集合通信是分布式训练的基础。它描述的是一组 rank 共同参与的数据移动操作。

这一章的第一目标不是死记 API，而是建立直觉：每个操作会如何改变各个 rank 上的数据。

## 核心词汇

- Rank：分布式任务中的一个参与者。
- World size：某个 process group 中 rank 的总数。
- Process group：可以相互通信的一组 rank。
- Backend：真正执行通信的后端实现，例如 NCCL 或 Gloo。
- Tensor：PyTorch 分布式 API 中最常被通信的数据单位。
- Collective：group 内所有 rank 都需要共同调用的通信操作。

## 心智模型

大多数 collective 都是在回答四类问题：

- 是否要让一个 rank 把数据发给所有 rank？
- 是否要让所有 rank 贡献数据，并在一个 rank 上得到结果？
- 是否要让所有 rank 都得到同一个聚合结果？
- 是否要让所有 rank 交换不同的数据片段？

第一张地图可以这样看：

| 操作 | 核心含义 | 常见训练用途 |
| --- | --- | --- |
| `broadcast` | 一个 rank 把同一个 tensor 发给所有 rank | 同步初始参数 |
| `reduce` | 所有 rank 贡献数据，一个 rank 得到结果 | 指标聚合或集中式统计 |
| `all_reduce` | 所有 rank 贡献数据，所有 rank 得到结果 | DDP 梯度同步 |
| `all_gather` | 所有 rank 把本地 tensor 分享给所有 rank | 重建 sharded tensor |
| `reduce_scatter` | 先 reduce，再把结果切片分给各 rank | ZeRO/FSDP 风格的梯度切分 |
| `all_to_all` | 每个 rank 向其他 rank 发送不同数据块 | MoE token dispatch |
| `barrier` | 所有 rank 等待彼此到达同一点 | 调试与同步 |

## 学习顺序

1. [基本 collective](01-basic-collectives.md)
2. [通信代价模型](02-communication-cost-model.md)
3. [Ring All-Reduce](03-ring-all-reduce.md)
4. Tree 与 hierarchical collective
5. [集合通信到训练的映射](04-training-mapping.md)
6. 最小 PyTorch 实验

## 第一组阅读问题

在接触 NCCL 或 PyTorch 源码之前，我们先回答这些问题：

- 如果每个 rank 都有一个梯度 tensor，为什么 DDP 需要 `all_reduce`？
- 为什么 `all_reduce` 经常可以理解成 `reduce_scatter` 加 `all_gather`？
- 为什么通信时间同时依赖 latency 和 bandwidth？
- 为什么 API 相同，但硬件 topology 不同会导致性能差异？
- 为什么两个都正确的 collective 算法可能性能完全不同？

## 本章产出

学完本章后，我们应该得到：

- 一套能读分布式训练代码的基础词汇。
- 一张训练行为到 collective primitive 的映射表。
- 一个能粗略分析性能的通信代价模型。
- 对 ring all-reduce 的完整解释。
- 每个重要 primitive 的最小 PyTorch 实验。
