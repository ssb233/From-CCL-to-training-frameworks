# NCCL

This chapter studies NCCL as the GPU collective communication backend.

Initial questions:

- What is an NCCL communicator?
- How does NCCL choose rings, trees, channels, and protocols?
- What happens inside `ncclAllReduce`?
- How do CUDA streams interact with communication?
- How do topology and environment variables affect performance?

Source reading targets will be added after we pin an NCCL version in
[versions.md](../../references/versions.md).

