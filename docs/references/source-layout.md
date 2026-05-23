# Source Layout Strategy

The learning repository should contain our notes, experiments, diagrams, and
small code snippets. Full framework repositories should usually live outside the
learning repository.

Recommended layout:

```text
Desktop/
  study/
  pytorch/
  nccl/
  Megatron-LM/
  DeepSpeed/
```

Why:

- The learning repository stays small and readable.
- Source repositories keep their own Git history.
- We can pin exact commits without copying large codebases.
- Notes can link to local paths and upstream GitHub URLs.

When source reading begins, each note should record:

- Project name
- Commit hash or release tag
- File path
- Function or class name
- Why the code matters
- What question the code answers

