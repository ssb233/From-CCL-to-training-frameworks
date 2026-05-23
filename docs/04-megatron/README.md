# Megatron-LM

This chapter studies Megatron-LM as a reference implementation of large-model
parallel training.

Initial questions:

- How are tensor, pipeline, and data parallel process groups created?
- Why do column-parallel and row-parallel linear layers need different
  collectives?
- How does sequence parallelism change activation and gradient communication?
- What does one pipeline-parallel training step do across microbatches?
- How does the distributed optimizer reduce memory?

Source reading targets will be added after we pin a Megatron-LM commit in
[versions.md](../../references/versions.md).

