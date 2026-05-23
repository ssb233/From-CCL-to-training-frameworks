# 集合通信到训练的映射

Collective 不是抽象数学题。每个 collective 出现，都是因为某种训练策略制造了特定的数据移动问题。

## 数据并行训练

在数据并行中：

- 每个 rank 有完整模型副本。
- 每个 rank 处理不同输入数据。
- 每个 rank 计算本地梯度。
- 模型副本必须保持数值同步。

核心 collective 是：

```text
all_reduce(local_gradients)
```

all-reduce 并平均后，每个 rank 执行相同的 optimizer update。

## 张量并行训练

在张量并行中：

- 一个 layer 被拆到多个 rank 上。
- 每个 rank 计算该 layer 输出或梯度的一部分。
- 某些 layer 边界需要合并 partial result。

常见 collective：

- `all_reduce`
- `all_gather`
- `reduce_scatter`

直觉示例：

- 如果各 rank 计算的是 partial sum，需要 `all_reduce`。
- 如果各 rank 持有 shard，但后续需要完整 tensor，需要 `all_gather`。
- 如果各 rank 算出了完整结果，但最终只应该保留 shard，需要 `reduce_scatter`。

## 流水线并行训练

在流水线并行中：

- 不同 rank 持有不同 layer 或 block。
- activation 沿 forward 方向在 stage 间移动。
- gradient 沿 backward 方向在 stage 间移动。

这类通信经常使用点对点通信，而不只是 collective。

重要结论：

```text
并不是所有分布式训练通信都是 collective
```

流水线并行会迫使我们后面研究 send/recv。

## ZeRO 与 Sharded Training

在 ZeRO 风格训练中：

- optimizer state 可能被切分。
- gradient 可能被切分。
- parameter 可能被切分。

常见 collective：

- `reduce_scatter`：reduce 梯度并只保留本 rank shard。
- `all_gather`：在需要时临时 materialize 参数。

核心 tradeoff：

```text
更少的显存复制，换来更结构化的通信
```

## Mixture-of-Experts

在 expert parallelism 中：

- token 被路由到不同 expert rank。
- 每个 rank 可能向其他 rank 发送不同 token 子集。

常见 collective 是：

```text
all_to_all
```

这种通信模式更难负载均衡，因为 token routing 可能不均匀。

## 总结表

| 训练模式 | 数据移动问题 | 常见通信 |
| --- | --- | --- |
| DDP | 跨模型副本平均梯度 | `all_reduce` |
| FSDP / ZeRO | reduce 并切分梯度 | `reduce_scatter` |
| FSDP / ZeRO | materialize sharded parameter | `all_gather` |
| Tensor parallel linear layer | 合并 partial result | `all_reduce` 或 `all_gather` |
| Pipeline parallelism | 在 stage 间移动 activation 和 gradient | send/recv |
| Expert parallelism | 把 token 路由给 expert | `all_to_all` |

## 源码阅读时的提问方式

读框架源码时，始终问：

1. 当前 rank 上有什么 tensor？
2. 接下来哪些 rank 需要这个 tensor？
3. 这个值应该被复制、reduce、scatter，还是 exchange？
4. 这个通信在 critical path 上吗？能否和 compute overlap？
