# Glossary

## Rank

One participant in a distributed job. In PyTorch, each rank is usually one
process.

## World Size

The number of ranks in a process group.

## Process Group

A subset of ranks that can communicate with each other.

## Backend

The implementation used to run distributed communication. Common PyTorch
backends include NCCL and Gloo.

## Collective Communication

A communication operation that every rank in a group calls together.

## Data Parallelism

Each rank owns a model replica and processes different data. Gradients are
communicated so replicas stay synchronized.

## Tensor Parallelism

Individual model layers are split across ranks. Communication happens inside the
forward and backward pass of a layer.

## Pipeline Parallelism

Different layers or blocks are placed on different ranks. Microbatches flow
through pipeline stages.

## ZeRO

DeepSpeed's optimizer-state, gradient, and parameter partitioning family of
techniques for reducing memory replication.

