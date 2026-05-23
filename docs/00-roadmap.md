# 学习路线图

这条路线不是按照框架名字来切分，而是按照分布式训练真正的执行链路来组织。

## 阶段 1：集合通信

目标：理解后续所有训练框架都会依赖的通信原语。

主题：

- rank、world size、process group、backend
- 点对点通信与集合通信的区别
- `broadcast`、`reduce`、`all_reduce`、`all_gather`、`reduce_scatter`、`scatter`、`gather`、`all_to_all`、`barrier`
- latency、bandwidth、topology、synchronization
- ring、tree、hierarchical、topology-aware 算法
- 为什么梯度同步经常对应 `all_reduce`
- 为什么张量并行经常使用 `all_reduce`、`all_gather` 和 `reduce_scatter`

产出：

- 每个通信原语的笔记
- 小型 PyTorch distributed 实验
- 通信模式图

## 阶段 2：PyTorch Distributed

目标：把我们写的 PyTorch API 和底层运行时行为连起来。

主题：

- `torch.distributed.init_process_group`
- store、rendezvous、rank 分配、launcher
- process group 和 subgroup
- `ProcessGroupNCCL`
- DDP gradient bucket 和通信计算重叠
- FSDP 与 `reduce_scatter` / `all_gather` 风格的 sharding

产出：

- 最小可运行实验
- API 到源码的调用链
- DDP 与 FSDP 通信模式对比

## 阶段 3：NCCL

目标：理解 NCCL 作为 GPU 通信引擎的工作方式。

主题：

- communicator、rank、channel、stream
- ring、tree、protocol、topology
- `ncclAllReduce`、`ncclBroadcast`、`ncclReduceScatter`、`ncclAllGather`
- 单机内 NVLink / PCIe 与多机网络通信
- 使用 `NCCL_DEBUG`、`NCCL_TOPO_DUMP_FILE` 和环境变量调试

产出：

- NCCL 概念图
- 关键函数源码阅读笔记
- NCCL 调试清单

## 阶段 4：Megatron-LM

目标：理解大模型并行训练策略如何组合。

主题：

- 数据并行、张量并行、流水线并行
- model-parallel process group 的构建
- ColumnParallelLinear 与 RowParallelLinear
- sequence parallelism
- pipeline schedule 与 microbatch
- distributed optimizer

产出：

- process group 图
- layer 级通信笔记
- 一次训练 step 的完整 walkthrough

## 阶段 5：DeepSpeed

目标：理解 DeepSpeed 作为训练引擎和显存优化系统的设计。

主题：

- engine 初始化和训练循环包装
- ZeRO stage 1、2、3
- parameter、gradient、optimizer state 的切分
- communication hook 与 overlap
- checkpoint 与 offload
- 与 Megatron 风格训练的集成

产出：

- ZeRO 通信与显存表
- DeepSpeed engine 调用链
- 与 PyTorch DDP/FSDP 和 Megatron distributed optimizer 的对比

## 阶段 6：端到端训练流程

目标：解释一次完整的分布式训练迭代。

主题：

- 启动进程
- 初始化 process group
- 构建模型
- 包装并行策略
- 加载数据
- forward
- loss 计算
- backward
- 梯度、参数或优化器状态通信
- optimizer step
- checkpoint 与 logging

产出：

- 完整 training step 图
- 最小 PyTorch distributed training demo
- PyTorch DDP、Megatron-LM、DeepSpeed 流程对比

## 学习方法

每个主题都按同一个结构沉淀：

1. 概念：它解决什么问题？
2. API：用户写什么代码？
3. 运行时：各个 rank 上发生了什么？
4. 源码：实现在哪里？
5. 实验：如何验证？
6. 总结：应该记住什么？
