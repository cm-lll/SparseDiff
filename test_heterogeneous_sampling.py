#!/usr/bin/env python3
"""
测试异质图采样过程中的状态空间映射
"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')

print("=" * 80)
print("测试异质图采样过程中的状态空间映射")
print("=" * 80)

# 模拟异质图数据
print("\n1. 创建模拟数据...")
num_edges = 10
num_global_states = 6  # 全局状态空间：0 (no-edge), 1-3 (fam_A), 4-5 (fam_B)

# 模拟边属性（全局ID）
comp_edge_attr_discrete = torch.tensor([0, 1, 2, 3, 0, 4, 5, 0, 1, 4])  # 混合了关系族A和B的边
comp_edge_attr = F.one_hot(comp_edge_attr_discrete, num_classes=num_global_states).float()

# 模拟关系族信息
edge_family_offsets = {
    "paper-cites-paper": 1,  # offset=1, 全局ID 1-3
    "author-writes-paper": 4,  # offset=4, 全局ID 4-5
}
edge_family2id = {
    "paper-cites-paper": 0,
    "author-writes-paper": 1,
}
id2edge_family = {v: k for k, v in edge_family2id.items()}

print(f"  全局边属性ID: {comp_edge_attr_discrete.tolist()}")
print(f"  关系族A (paper-cites-paper): offset={edge_family_offsets['paper-cites-paper']}, 全局ID 1-3")
print(f"  关系族B (author-writes-paper): offset={edge_family_offsets['author-writes-paper']}, 全局ID 4-5")

# 模拟转移矩阵
print("\n2. 创建模拟转移矩阵...")
device = torch.device("cpu")
batch_size = 1

# 关系族A：4个状态 (0=no-edge, 1-3=子类型)
num_fam_A_states = 4
Qt_fam_A = torch.eye(num_fam_A_states).unsqueeze(0).expand(batch_size, -1, -1)  # (1, 4, 4)
Qsb_fam_A = torch.eye(num_fam_A_states).unsqueeze(0).expand(batch_size, -1, -1)  # (1, 4, 4)
Qtb_fam_A = torch.eye(num_fam_A_states).unsqueeze(0).expand(batch_size, -1, -1)  # (1, 4, 4)

# 关系族B：3个状态 (0=no-edge, 1-2=子类型)
num_fam_B_states = 3
Qt_fam_B = torch.eye(num_fam_B_states).unsqueeze(0).expand(batch_size, -1, -1)  # (1, 3, 3)
Qsb_fam_B = torch.eye(num_fam_B_states).unsqueeze(0).expand(batch_size, -1, -1)  # (1, 3, 3)
Qtb_fam_B = torch.eye(num_fam_B_states).unsqueeze(0).expand(batch_size, -1, -1)  # (1, 3, 3)

print(f"  关系族A转移矩阵: {Qt_fam_A.shape}")
print(f"  关系族B转移矩阵: {Qt_fam_B.shape}")

# 模拟批次信息
batch = torch.zeros(num_edges, dtype=torch.long)

# 步骤3：为每个关系族计算转移概率
print("\n3. 为每个关系族计算转移概率...")
p_s_and_t_given_0_E_list = []

for fam_id, fam_name in id2edge_family.items():
    offset = edge_family_offsets[fam_name]
    
    # 判断哪些边属于这个关系族
    if fam_name == "paper-cites-paper":
        # 关系族A：全局ID 0 或 1-3
        next_offset = edge_family_offsets["author-writes-paper"]
        fam_mask = (comp_edge_attr_discrete == 0) | ((comp_edge_attr_discrete >= offset) & (comp_edge_attr_discrete < next_offset))
        num_fam_states = num_fam_A_states
        Qt_fam = Qt_fam_A
        Qsb_fam = Qsb_fam_A
        Qtb_fam = Qtb_fam_A
    else:
        # 关系族B：全局ID 0 或 4-5
        fam_mask = (comp_edge_attr_discrete == 0) | ((comp_edge_attr_discrete >= offset) & (comp_edge_attr_discrete < num_global_states))
        num_fam_states = num_fam_B_states
        Qt_fam = Qt_fam_B
        Qsb_fam = Qsb_fam_B
        Qtb_fam = Qtb_fam_B
    
    if not fam_mask.any():
        continue
    
    print(f"\n  处理关系族: {fam_name} (offset={offset})")
    print(f"    属于该关系族的边索引: {torch.where(fam_mask)[0].tolist()}")
    
    # 获取该关系族的边
    fam_comp_edge_attr_discrete = comp_edge_attr_discrete[fam_mask]
    print(f"    全局ID: {fam_comp_edge_attr_discrete.tolist()}")
    
    # 转换为局部ID
    fam_local_attr = fam_comp_edge_attr_discrete.clone()
    non_zero_mask = fam_local_attr != 0
    if non_zero_mask.any():
        fam_local_attr[non_zero_mask] = fam_local_attr[non_zero_mask] - offset + 1
    print(f"    局部ID: {fam_local_attr.tolist()}")
    
    # 转换为 one-hot 编码
    fam_local_attr_onehot = F.one_hot(fam_local_attr.long(), num_classes=num_fam_states).float()
    print(f"    局部 one-hot 形状: {fam_local_attr_onehot.shape}")
    
    # 模拟计算转移概率（简化版本）
    num_edges_fam = fam_local_attr_onehot.shape[0]
    # 使用简化的后验分布计算
    p_s_and_t_given_0_E_fam = torch.ones(
        (num_edges_fam, num_fam_states, num_fam_states)
    ) / num_fam_states  # 简化为均匀分布
    
    print(f"    转移概率形状: {p_s_and_t_given_0_E_fam.shape}")
    p_s_and_t_given_0_E_list.append((fam_mask, p_s_and_t_given_0_E_fam, offset, num_fam_states))

# 步骤4：映射到全局状态空间
print("\n4. 映射到全局状态空间...")
p_s_and_t_given_0_E = torch.zeros(
    (num_edges, num_global_states, num_global_states),
    device=device
)

for fam_mask, p_s_and_t_given_0_E_fam, offset, num_fam_states in p_s_and_t_given_0_E_list:
    num_edges_fam = p_s_and_t_given_0_E_fam.shape[0]
    
    print(f"\n  映射关系族 (offset={offset}, num_states={num_fam_states}):")
    
    # 创建映射：从局部状态到全局状态
    for local_from in range(num_fam_states):
        if local_from == 0:
            global_from = 0
        else:
            global_from = offset + local_from - 1
        
        for local_to in range(num_fam_states):
            if local_to == 0:
                global_to = 0
            else:
                global_to = offset + local_to - 1
            
            if global_from < num_global_states and global_to < num_global_states:
                # 复制转移概率
                p_s_and_t_given_0_E[fam_mask, global_from, global_to] = \
                    p_s_and_t_given_0_E_fam[:, local_from, local_to]
                print(f"    局部 ({local_from}, {local_to}) -> 全局 ({global_from}, {global_to})")

# 验证结果
print("\n5. 验证结果...")
print(f"  全局转移概率矩阵形状: {p_s_and_t_given_0_E.shape}")
print(f"  每个边的转移概率矩阵形状: {p_s_and_t_given_0_E[0].shape}")

# 检查第一个边（应该是 no-edge，全局ID=0）
print(f"\n  第一个边（全局ID=0）的转移概率矩阵:")
print(f"    {p_s_and_t_given_0_E[0]}")

# 检查第二个边（应该是关系族A，全局ID=1）
print(f"\n  第二个边（全局ID=1，关系族A）的转移概率矩阵:")
print(f"    {p_s_and_t_given_0_E[1]}")

# 检查第六个边（应该是关系族B，全局ID=4）
print(f"\n  第六个边（全局ID=4，关系族B）的转移概率矩阵:")
print(f"    {p_s_and_t_given_0_E[5]}")

print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)
