# 从集合通信到训练框架

这个仓库用于系统性记录从集合通信到分布式深度学习训练框架的学习过程。

学习的主要入口是对话：先通过聊天完成讲解、追问、例子和理解检查，再把结论整理到文档中。仓库文档主要用于课后复习、源码索引、实验记录和长期进度追踪。

核心技术栈包括：

- PyTorch：训练代码和分布式 API 的基础
- `torch.distributed`：PyTorch 分布式运行时入口
- NCCL：GPU 集合通信后端
- Megatron-LM：张量并行、流水线并行、数据并行的大模型训练参考实现
- DeepSpeed：ZeRO、训练引擎和显存优化参考实现

## 学习目标

建立一条完整的理解链路：从 `all_reduce` 这样的单个集合通信原语开始，一直理解到一个完整的大模型分布式训练 step。

这个仓库里的笔记应该逐步回答这些问题：

- 每个集合通信原语解决什么问题？
- PyTorch 如何把一次 collective 调用路由到 NCCL？
- 数据并行、张量并行、流水线并行、ZeRO 分别如何改变通信模式？
- Megatron-LM 和 DeepSpeed 如何组织 process group、梯度、优化器状态和 checkpoint？
- 一次端到端训练迭代到底发生了什么？

## 仓库结构

```text
docs/
  00-roadmap.md
  01-collective-communication/
  02-pytorch-distributed/
  03-nccl/
  04-megatron/
  05-deepspeed/
  06-end-to-end-training-loop/
  references/
    versions.md
    source-layout.md
code-snippets/
diagrams/
sessions/
progress/
```

这个仓库应保持为“学习资料仓库”，不默认复制完整上游源码。完整源码更适合 clone 到这个仓库旁边。

推荐本地布局：

```text
Desktop/
  study/
  pytorch/
  nccl/
  Megatron-LM/
  DeepSpeed/
```

## 从这里开始

1. 先读 [学习路线图](docs/00-roadmap.md)。
2. 从 [集合通信](docs/01-collective-communication/README.md) 开始。
3. 每次阅读源码时，把版本记录到 [源码版本记录](docs/references/versions.md)。

## 引导式学习命令

如果使用 Claude Code 风格的命令，可以使用：

```text
/dist-train-study <主题或问题>
```

命令定义位于：

[.claude/commands/dist-train-study.md](.claude/commands/dist-train-study.md)

长期学习进度记录位于：

[progress/distributed-training-study-tracker.md](progress/distributed-training-study-tracker.md)
