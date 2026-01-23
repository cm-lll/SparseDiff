# 内存问题检查

## 问题分析

从错误信息看，OOM 发生在：
```python
adj_matrix = noisy_data["E_t"][..., 1:].int().sum(dim=-1)  # (bs, n, n)
```

这是在 `densify_noisy_data` 之后，`E_t` 变成了密集格式 `(bs, n, n, de)`。

## 可能的问题

### 1. 损失计算中的内存占用

在 `compute_Lt` 中，我创建了：
- `prob_true_E = torch.zeros_like(E)` - (bs, n, n, de)
- `prob_pred_E = torch.zeros_like(pred_probs_E)` - (bs, n, n, de)

对于大图（n很大），这些张量会占用大量内存。

### 2. 验证步骤中的密集化

在 `validation_step` 中：
- `densify_noisy_data` 将稀疏数据转换为密集格式
- 对于大图，这会占用大量内存

### 3. 中间张量未释放

在损失计算中，创建了很多中间张量：
- one-hot 编码
- 局部状态张量
- 映射后的全局状态张量

## 优化建议

1. **使用 inplace 操作**：减少内存分配
2. **及时释放中间张量**：使用 `del` 或 `torch.cuda.empty_cache()`
3. **减小 batch_size**：最简单有效的方法
4. **优化验证步骤**：避免不必要的密集化

