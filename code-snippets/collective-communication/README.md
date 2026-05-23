# 集合通信实验

这个目录用于存放最小 PyTorch distributed 实验。

计划脚本：

- `all_reduce_demo.py`
- `all_gather_demo.py`
- `reduce_scatter_demo.py`
- `broadcast_demo.py`

预期启动方式：

```powershell
torchrun --nproc_per_node=4 code-snippets/collective-communication/all_reduce_demo.py
```

每个实验都只回答一个小问题：

- 每个 rank 开始时有什么 tensor？
- 调用了哪个 collective？
- 每个 rank 结束时有什么 tensor？
