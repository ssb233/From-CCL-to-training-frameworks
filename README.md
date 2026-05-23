# From CCL to Training Frameworks

This repository records a systematic learning path from collective communication
to distributed deep learning training frameworks.

The core stack is:

- PyTorch as the programming and training foundation
- `torch.distributed` as the distributed runtime entry point
- NCCL as the GPU collective communication backend
- Megatron-LM as the tensor/pipeline/data parallel training reference
- DeepSpeed as the ZeRO and training engine reference

## Learning Goal

Build a complete mental model of how distributed training works, from a single
collective primitive such as `all_reduce` to a full large-model training loop.

The notes should answer questions like:

- What problem does each collective communication primitive solve?
- How does PyTorch route a collective call to NCCL?
- How do data parallel, tensor parallel, pipeline parallel, and ZeRO change
  communication patterns?
- How do Megatron-LM and DeepSpeed organize process groups, gradients,
  optimizer states, and checkpoints?
- What happens during one end-to-end training step?

## Repository Layout

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
```

The learning repository should stay clean. Full upstream source repositories are
better cloned next to this repository or added later as pinned references only
when source-level reading begins.

Recommended local layout:

```text
Desktop/
  study/
  pytorch/
  nccl/
  Megatron-LM/
  DeepSpeed/
```

## Start Here

1. Read [the roadmap](docs/00-roadmap.md).
2. Start with [collective communication](docs/01-collective-communication/README.md).
3. Record source versions in [versions.md](docs/references/versions.md) whenever we
   read framework code.
