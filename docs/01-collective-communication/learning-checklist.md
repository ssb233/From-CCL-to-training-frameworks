# 学习清单

用这份清单追踪集合通信章节的学习进度。

## 概念

- [x] 理解 rank、world size、process group 和 backend。
- [x] 区分点对点通信和集合通信。
- [x] 解释 `broadcast`。
- [x] 解释 `reduce`。
- [x] 解释 `all_reduce`。
- [x] 解释 `all_gather`。
- [x] 解释 `reduce_scatter`。
- [x] 解释 `all_to_all`。
- [x] 解释 `barrier`。
- [x] 解释 latency 和 bandwidth。
- [x] 解释为什么 bucketing 对 DDP 有帮助。
- [x] 从高层解释 ring all-reduce。
- [ ] 解释 tree collective。
- [ ] 解释 hierarchical collective。

## 训练映射

- [x] 把 DDP 梯度同步映射到 `all_reduce`。
- [x] 把 ZeRO/FSDP 梯度切分映射到 `reduce_scatter`。
- [x] 把 ZeRO/FSDP 参数 materialization 映射到 `all_gather`。
- [x] 把 expert routing 映射到 `all_to_all`。
- [ ] 把 Megatron tensor-parallel linear layer 映射到具体 collective。

## 实验

- [ ] 运行本地 `all_reduce` 示例。
- [ ] 运行本地 `all_gather` 示例。
- [ ] 运行本地 `reduce_scatter` 示例。
- [ ] 检查每个 rank 上 tensor 值如何变化。
- [ ] 为核心 collective 补图。
