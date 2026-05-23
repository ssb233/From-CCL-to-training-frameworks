# Megatron-LM

本章研究 Megatron-LM 作为大模型并行训练参考实现的设计。

初始问题：

- tensor parallel、pipeline parallel、data parallel process group 如何创建？
- 为什么 ColumnParallelLinear 和 RowParallelLinear 需要不同 collective？
- sequence parallelism 如何改变 activation 和 gradient 通信？
- 一次 pipeline-parallel training step 如何跨 microbatch 执行？
- distributed optimizer 如何降低显存占用？

源码阅读目标会在 [源码版本记录](../references/versions.md) 中固定 Megatron-LM commit 后继续补充。
