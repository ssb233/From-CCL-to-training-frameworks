# PyTorch Distributed

本章把用户侧 PyTorch API 和分布式运行时连接起来。

初始问题：

- `init_process_group` 到底初始化了什么？
- rank 和 world size 如何分配？
- process group 是什么？
- 一次 `all_reduce` 调用如何走到 `ProcessGroupNCCL`？
- DDP 如何组织 gradient bucket，并把通信与 backward 重叠？
- FSDP 如何使用 sharding 风格的 collective？

源码阅读目标会在 [源码版本记录](../references/versions.md) 中固定 PyTorch 版本后继续补充。
