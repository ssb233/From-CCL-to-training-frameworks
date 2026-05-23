# End-to-End Training Loop

This chapter ties the previous chapters together.

The target is to explain one complete distributed training iteration:

1. Launch distributed processes.
2. Initialize process groups.
3. Construct model and parallel groups.
4. Load data shards.
5. Run forward pass.
6. Compute loss.
7. Run backward pass.
8. Communicate gradients, parameters, or optimizer shards.
9. Step optimizer.
10. Save checkpoint and log metrics.

We will compare three flows:

- Plain PyTorch DDP
- Megatron-LM style model-parallel training
- DeepSpeed ZeRO training

