# 端到端训练流程

本章把前面所有章节串起来。

目标是解释一次完整的分布式训练迭代：

1. 启动分布式进程。
2. 初始化 process group。
3. 构建模型和并行组。
4. 加载数据 shard。
5. 执行 forward。
6. 计算 loss。
7. 执行 backward。
8. 通信 gradient、parameter 或 optimizer shard。
9. 执行 optimizer step。
10. 保存 checkpoint 并记录指标。

我们会对比三条流程：

- 纯 PyTorch DDP
- Megatron-LM 风格的 model-parallel training
- DeepSpeed ZeRO training
