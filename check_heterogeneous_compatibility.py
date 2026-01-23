#!/usr/bin/env python3
"""
检查去噪模型中不适配异质图的部分
"""
import sys

sys.path.insert(0, '.')

# 检查清单
issues = []

print("=" * 80)
print("检查去噪模型中不适配异质图的部分")
print("=" * 80)

# 1. 采样过程中的转移矩阵使用
print("\n1. 采样过程 (sample_p_zs_given_zt, 1586行)")
print("   - 第1604-1606行：使用 get_Qt_bar 和 get_Qt 时没有考虑关系族隔离")
print("   - 第1728-1736行：计算 p_s_and_t_given_0_E 时使用了全局的 Qt.E, Qsb.E, Qtb.E")
print("   ⚠️  问题：对于异质图，应该为每个关系族使用不同的转移矩阵")
issues.append({
    "location": "sample_p_zs_given_zt (1586行)",
    "issue": "使用全局转移矩阵，没有考虑关系族隔离",
    "lines": [1604, 1605, 1606, 1728, 1732, 1733, 1734]
})

# 2. 损失计算中的转移矩阵使用
print("\n2. 损失计算 (compute_val_loss, 1053行 和 compute_Lt, 1187行)")
print("   - 第1108行：Qtb = self.transition_model.get_Qt_bar(...) - 没有考虑关系族隔离")
print("   - 第1112行：probE = E @ Qtb.E.unsqueeze(1) - 使用全局转移矩阵")
print("   - 第1197-1199行：同样的问题")
print("   ⚠️  问题：对于异质图，应该为每个关系族使用不同的转移矩阵计算损失")
issues.append({
    "location": "compute_val_loss / compute_Lt (1053行, 1187行)",
    "issue": "使用全局转移矩阵计算损失，没有考虑关系族隔离",
    "lines": [1108, 1112, 1197, 1198, 1199]
})

# 3. kl_prior 中的转移矩阵使用
print("\n3. KL 先验计算 (kl_prior, 1098行)")
print("   - 第1108行：Qtb = self.transition_model.get_Qt_bar(...) - 没有考虑关系族隔离")
print("   - 第1112行：probE = E @ Qtb.E.unsqueeze(1) - 使用全局转移矩阵")
print("   ⚠️  问题：对于异质图，应该为每个关系族使用不同的转移矩阵")
issues.append({
    "location": "kl_prior (1098行)",
    "issue": "使用全局转移矩阵计算 KL 先验，没有考虑关系族隔离",
    "lines": [1108, 1112]
})

# 4. 采样边时的转移矩阵使用
print("\n4. 采样边 (sample_sparse_edge, 1549行)")
print("   - 使用 p_s_and_t_given_0_E，这个需要检查是否正确考虑了关系族隔离")
print("   - p_s_and_t_given_0_E 在 sample_p_zs_given_zt 中计算，使用了全局转移矩阵")
print("   ⚠️  问题：采样时应该为每个关系族使用不同的转移矩阵")
issues.append({
    "location": "sample_sparse_edge (1549行)",
    "issue": "依赖的 p_s_and_t_given_0_E 使用了全局转移矩阵",
    "lines": [1549, 1728]
})

print("\n" + "=" * 80)
print("总结")
print("=" * 80)
print(f"发现 {len(issues)} 个不适配异质图的地方：\n")

for i, issue in enumerate(issues, 1):
    print(f"{i}. {issue['location']}")
    print(f"   问题：{issue['issue']}")
    print(f"   相关行：{issue['lines']}")
    print()

print("=" * 80)
print("建议的修复方案")
print("=" * 80)
print("""
1. **采样过程 (sample_p_zs_given_zt)**：
   - 需要为每个关系族计算独立的 p_s_and_t_given_0_E
   - 在采样边时，需要根据 edge_family 选择对应的转移矩阵

2. **损失计算 (compute_val_loss / compute_Lt)**：
   - 需要为每个关系族计算独立的损失
   - 或者使用全局转移矩阵作为近似（但可能不够准确）

3. **KL 先验 (kl_prior)**：
   - 需要为每个关系族计算独立的 KL 先验
   - 或者使用全局转移矩阵作为近似

4. **采样边 (sample_sparse_edge)**：
   - 需要根据 edge_family 选择对应的 p_s_and_t_given_0_E
   - 或者修改 sample_p_zs_given_zt 使其返回按关系族分组的转移概率
""")
