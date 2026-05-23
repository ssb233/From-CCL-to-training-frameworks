# DeepSpeed

This chapter studies DeepSpeed as a training engine and memory optimization
system.

Initial questions:

- What does `deepspeed.initialize` construct?
- How does the DeepSpeed engine wrap the training loop?
- What changes from ZeRO stage 1 to stage 2 to stage 3?
- Which tensors are partitioned: optimizer states, gradients, parameters?
- Where do `reduce_scatter` and `all_gather` appear in ZeRO?
- How does DeepSpeed integrate with Megatron-style model parallelism?

Source reading targets will be added after we pin a DeepSpeed version in
[versions.md](../../references/versions.md).

