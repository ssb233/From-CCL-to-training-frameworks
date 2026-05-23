# DeepSpeed

本章研究 DeepSpeed 作为训练引擎和显存优化系统的设计。

初始问题：

- `deepspeed.initialize` 构造了什么？
- DeepSpeed engine 如何包装训练循环？
- ZeRO stage 1、2、3 分别改变了什么？
- 哪些 tensor 被切分：optimizer state、gradient、parameter？
- ZeRO 中 `reduce_scatter` 和 `all_gather` 出现在哪里？
- DeepSpeed 如何与 Megatron 风格的 model parallelism 集成？

源码阅读目标会在 [源码版本记录](../references/versions.md) 中固定 DeepSpeed 版本后继续补充。
