# 术语表

## Rank

分布式任务中的一个参与者。在 PyTorch 中，一个 rank 通常对应一个进程。

## World Size

某个 process group 中 rank 的数量。

## Process Group

一组可以相互通信的 rank。

## Backend

执行分布式通信的后端实现。PyTorch 中常见后端包括 NCCL 和 Gloo。

## Collective Communication

一个 group 内所有 rank 共同调用的通信操作。

## Data Parallelism

每个 rank 持有一份模型副本，并处理不同数据。各 rank 通过通信同步梯度，使模型副本保持一致。

## Tensor Parallelism

把单个模型 layer 切分到多个 rank 上。通信通常发生在 layer 的 forward 和 backward 内部。

## Pipeline Parallelism

把不同 layer 或 block 放到不同 rank 上。microbatch 会沿 pipeline stage 流动。

## ZeRO

DeepSpeed 的一组显存优化技术，通过切分 optimizer state、gradient 和 parameter 来减少复制。
