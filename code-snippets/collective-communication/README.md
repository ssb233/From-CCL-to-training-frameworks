# Collective Communication Experiments

This directory will hold minimal PyTorch distributed experiments.

Planned scripts:

- `all_reduce_demo.py`
- `all_gather_demo.py`
- `reduce_scatter_demo.py`
- `broadcast_demo.py`

Expected launch shape:

```powershell
torchrun --nproc_per_node=4 code-snippets/collective-communication/all_reduce_demo.py
```

The experiments should stay small and answer one question each:

- What tensor does each rank start with?
- What collective is called?
- What tensor does each rank end with?

