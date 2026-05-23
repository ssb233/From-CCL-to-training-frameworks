# NCCL

本章研究 NCCL 作为 GPU 集合通信后端的工作方式。

初始问题：

- NCCL communicator 是什么？
- NCCL 如何选择 ring、tree、channel 和 protocol？
- `ncclAllReduce` 内部发生了什么？
- CUDA stream 如何与通信交互？
- topology 和环境变量如何影响性能？

源码阅读目标会在 [源码版本记录](../references/versions.md) 中固定 NCCL 版本后继续补充。
