# 源码目录策略

这个学习仓库应该存放我们的笔记、实验、图和小代码片段。完整框架源码通常放在学习仓库外部。

推荐布局：

```text
Desktop/
  study/
  pytorch/
  nccl/
  Megatron-LM/
  DeepSpeed/
```

原因：

- 学习仓库保持小而清晰。
- 源码仓库保留自己的 Git 历史。
- 我们可以固定精确 commit，而不复制大仓库。
- 笔记可以同时链接本地路径和上游 GitHub URL。

开始源码阅读时，每篇笔记都应记录：

- 项目名
- commit hash 或 release tag
- 文件路径
- 函数或类名
- 这段代码为什么重要
- 它回答了什么问题
