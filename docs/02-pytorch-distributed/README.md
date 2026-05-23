# PyTorch Distributed

This chapter connects user-facing PyTorch APIs to the distributed runtime.

Initial questions:

- What does `init_process_group` initialize?
- How are rank and world size assigned?
- What is a process group?
- How does an `all_reduce` call reach `ProcessGroupNCCL`?
- How does DDP bucket gradients and overlap communication with backward?
- How does FSDP use sharding-oriented collectives?

Source reading targets will be added after we pin a PyTorch version in
[versions.md](../references/versions.md).
