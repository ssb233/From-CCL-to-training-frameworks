# 分布式训练学习进度追踪

最后更新：2026-05-23

这个文件是长期学习进度的唯一事实来源。

## 章节进度

| 章节 | 状态 | 备注 |
| --- | --- | --- |
| 01 集合通信 | 进行中 | 已起草基本 collective、通信代价模型、ring all-reduce |
| 02 PyTorch Distributed | 未开始 | |
| 03 NCCL | 未开始 | |
| 04 Megatron-LM | 未开始 | |
| 05 DeepSpeed | 未开始 | |
| 06 端到端训练流程 | 未开始 | |

## 已掌握主题

| 主题 | 日期 | 信心 | 证据 |
| --- | --- | --- | --- |
| 学习仓库结构 | 2026-05-23 | 中高 | 已初始化 roadmap、docs、session 和 progress 结构 |

## 当前知识缺口

| 缺口 | 严重程度 | 关联章节 | 后续动作 |
| --- | --- | --- | --- |
| 需要动手运行 PyTorch collective 实验 | 中 | 01 集合通信 | 创建并运行 `all_reduce_demo.py` |
| 需要源码级理解 PyTorch API 到 NCCL backend 的路径 | 高 | 02 PyTorch Distributed | clone 或检查固定版本 PyTorch 源码 |

## 源码阅读状态

| 项目 | 状态 | 版本或 commit | 备注 |
| --- | --- | --- | --- |
| PyTorch | 未开始 | | |
| NCCL | 未开始 | | |
| Megatron-LM | 未开始 | | |
| DeepSpeed | 未开始 | | |

## 实验状态

| 实验 | 状态 | 备注 |
| --- | --- | --- |
| `all_reduce_demo.py` | 未开始 | 第一优先级 |
| `all_gather_demo.py` | 未开始 | |
| `reduce_scatter_demo.py` | 未开始 | |
| DDP 梯度同步 demo | 未开始 | |

## 下一步推荐主题

1. 运行本地 `all_reduce` 实验。
2. 为基本 collective 补图。
3. 学习 tree 和 hierarchical collective。
4. 开始 PyTorch distributed API 路径：`dist.all_reduce`。
