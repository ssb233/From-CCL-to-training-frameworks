# 通信代价模型

集合通信不只是“算得对”。同一个数学操作，即使结果正确，如果数据移动模式不好，也可能非常慢。

这篇笔记建立阅读 NCCL、PyTorch DDP、Megatron-LM 和 DeepSpeed 之前需要的第一个性能模型。

## 两类基本成本

一个简单通信模型会把时间拆成两部分：

```text
communication_time ~= latency_cost + bandwidth_cost
```

Latency 是发送消息的启动成本。Bandwidth 是搬运字节的成本。

常见形式是：

```text
time ~= alpha + n / beta
```

其中：

- `alpha` 是 latency。
- `n` 是字节数。
- `beta` 是 bandwidth，单位通常是 bytes/s。

这个模型很简单，但已经足以解释很多分布式训练行为。

## 小消息与大消息

对于小 tensor，latency 占主导：

```text
small n:
time ~= alpha
```

对于大 tensor，bandwidth 占主导：

```text
large n:
time ~= n / beta
```

这就是框架经常把很多小梯度合并成较大 bucket 的原因。一次较大的 all-reduce 通常比几千次很小的 all-reduce 更高效。

## Rank 数量的影响

增加 rank 可以减少每个 rank 的计算量，但通常也会增加通信协调成本。

更多 rank 可能意味着：

- 更多参与 collective 的进程。
- 更多网络链路被使用。
- 一个慢 rank 更容易拖住所有人。
- 更复杂的 topology 选择。

这是分布式训练无法完美线性扩展的第一个原因。

## Topology 很重要

API 会隐藏 topology：

```python
dist.all_reduce(tensor)
```

但运行时必须在真实硬件上选择数据路径。

单机内常见链路：

- NVLink
- PCIe
- GPU 到 CPU 内存路径

多机间常见链路：

- InfiniBand
- Ethernet
- RoCE

同一个 `all_reduce`，如果数据走 NVLink、PCIe 或跨节点网络，性能可能完全不同。

## 同步成本

Collective 需要 process group 中每个 rank 都调用。

如果一个 rank 到得晚，其他 rank 可能等待。

常见原因：

- 数据加载不均衡。
- 计算时间不均衡。
- 某个 GPU 或节点变慢。
- tensor shape 不一致或条件执行 bug。
- 网络拥塞。

所以通信性能也是系统问题，不只是算法问题。

## Bucketing

分布式训练框架经常在通信前把 tensor 组织成 bucket。

原因：

- 更少的 collective 调用可以降低 latency 开销。
- 更大的消息能更好利用 bandwidth。
- bucket 可以和 backward 计算重叠。

在 PyTorch DDP 中，梯度会被组织进 bucket。随着 backward 产生梯度，DDP 可以在某些 bucket ready 后立即发起 all-reduce，而不必等整个 backward 完成。

重要思想：

```text
通信不一定要等所有计算结束后才开始
```

优秀的训练框架会尽量让通信与计算重叠。

## 第一个扩展性直觉

假设每个 rank 都为同一个模型副本计算梯度。

增加 rank 会带来：

- 如果 global batch size 固定，每个 rank 处理的数据变少。
- 如果 per-rank batch size 固定，总 batch size 变大。
- 梯度同步需要跨更多 rank 完成。

训练 step 时间可以粗略理解为：

```text
step_time ~= compute_time + exposed_communication_time
```

这里的 `exposed` 很关键。有些通信可以藏在 backward 计算后面，有些不能。

## 对框架阅读的意义

| 框架区域 | 代价模型要问的问题 |
| --- | --- |
| PyTorch DDP | gradient bucket 如何形成并 reduce？ |
| NCCL | 后端选择了什么算法和 topology 路径？ |
| Megatron-LM | 哪些 layer 操作会在 forward/backward 中强制通信？ |
| DeepSpeed ZeRO | 哪些 tensor 被 partition，复制减少后由什么通信补回来？ |

## 关键结论

- 集合通信同时有 latency 和 bandwidth 成本。
- 小消息对 latency 敏感。
- 大消息对 bandwidth 敏感。
- bucketing 可以提升效率并支持 overlap。
- topology 会改变同一个 API 背后的真实成本。
- 分布式训练性能取决于有多少通信无法被计算隐藏。
