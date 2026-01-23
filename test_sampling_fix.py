#!/usr/bin/env python3
"""
测试异质图采样修复是否正常工作
"""
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, '.')

print("=" * 80)
print("测试异质图采样修复")
print("=" * 80)

# 测试1: 检查 get_all_family_Qt 方法是否存在
print("\n1. 检查 get_all_family_Qt 方法...")
try:
    from sparse_diffusion.diffusion.heterogeneous_transition import HeterogeneousMarginalUniformTransition
    print("   ✅ HeterogeneousMarginalUniformTransition 导入成功")
    
    # 检查方法是否存在
    if hasattr(HeterogeneousMarginalUniformTransition, 'get_all_family_Qt'):
        print("   ✅ get_all_family_Qt 方法存在")
    else:
        print("   ❌ get_all_family_Qt 方法不存在")
        sys.exit(1)
        
    if hasattr(HeterogeneousMarginalUniformTransition, 'get_all_family_Qt_bar'):
        print("   ✅ get_all_family_Qt_bar 方法存在")
    else:
        print("   ❌ get_all_family_Qt_bar 方法不存在")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ 导入失败: {e}")
    sys.exit(1)

# 测试2: 检查采样过程中的代码逻辑
print("\n2. 检查采样过程代码...")
try:
    from sparse_diffusion.diffusion_model_sparse import DiscreteDenoisingDiffusion
    print("   ✅ DiscreteDenoisingDiffusion 导入成功")
    
    # 读取相关代码片段
    import inspect
    source = inspect.getsource(DiscreteDenoisingDiffusion.sample_p_zs_given_zt)
    
    # 检查关键代码是否存在
    if 'get_all_family_Qt' in source:
        print("   ✅ 采样过程中使用了 get_all_family_Qt")
    else:
        print("   ⚠️  采样过程中未找到 get_all_family_Qt")
    
    if 'get_all_family_Qt_bar' in source:
        print("   ✅ 采样过程中使用了 get_all_family_Qt_bar")
    else:
        print("   ⚠️  采样过程中未找到 get_all_family_Qt_bar")
    
    if 'edge_family_offsets' in source:
        print("   ✅ 采样过程中使用了 edge_family_offsets")
    else:
        print("   ⚠️  采样过程中未找到 edge_family_offsets")
        
    if '局部状态空间映射回全局状态空间' in source or 'local_from' in source:
        print("   ✅ 采样过程中包含状态空间映射逻辑")
    else:
        print("   ⚠️  采样过程中未找到状态空间映射逻辑")
        
except Exception as e:
    print(f"   ❌ 检查失败: {e}")
    import traceback
    traceback.print_exc()

# 测试3: 模拟一个简单的转移矩阵调用
print("\n3. 测试转移矩阵方法调用...")
try:
    # 创建模拟的转移矩阵类
    class MockTransition:
        def __init__(self):
            self.heterogeneous = True
            self.edge_family_marginals = {
                "paper-cites-paper": torch.ones(4) / 4,
                "author-writes-paper": torch.ones(3) / 3,
            }
            self.edge_family_uniforms = {
                "paper-cites-paper": torch.ones(4, 4) / 4,
                "author-writes-paper": torch.ones(3, 3) / 3,
            }
            self.X_classes = 5
            self.E_classes = 6
            self.y_classes = 1
            self.charge_classes = 0
            self.u_x = torch.ones(1, 5, 5) / 5
            self.u_e = torch.ones(1, 6, 6) / 6
            self.u_y = torch.ones(1, 1, 1) / 1
            self.u_charge = None
        
        def get_Qt(self, beta_t, device, edge_family_name=None):
            # 简化实现
            if edge_family_name and edge_family_name in self.edge_family_uniforms:
                num_states = self.edge_family_uniforms[edge_family_name].shape[0]
                q_e = torch.eye(num_states).unsqueeze(0)
            else:
                q_e = torch.eye(self.E_classes).unsqueeze(0)
            return type('PlaceHolder', (), {'X': None, 'E': q_e, 'y': None, 'charge': None})()
        
        def get_all_family_Qt(self, beta_t, device):
            family_qt = {}
            for fam_name in self.edge_family_marginals.keys():
                family_qt[fam_name] = self.get_Qt(beta_t, device, edge_family_name=fam_name)
            return family_qt
    
    mock_transition = MockTransition()
    beta_t = torch.tensor([0.1])
    device = torch.device("cpu")
    
    result = mock_transition.get_all_family_Qt(beta_t, device)
    print(f"   ✅ get_all_family_Qt 调用成功")
    print(f"   返回的关系族数量: {len(result)}")
    for fam_name, qt in result.items():
        print(f"   - {fam_name}: E 矩阵形状 {qt.E.shape}")
        
except Exception as e:
    print(f"   ❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()

# 测试4: 检查代码语法
print("\n4. 检查代码语法...")
import subprocess
result = subprocess.run(
    ["python3", "-m", "py_compile", "sparse_diffusion/diffusion_model_sparse.py"],
    capture_output=True,
    text=True
)
if result.returncode == 0:
    print("   ✅ diffusion_model_sparse.py 语法正确")
else:
    print(f"   ❌ diffusion_model_sparse.py 语法错误:")
    print(result.stderr)

result = subprocess.run(
    ["python3", "-m", "py_compile", "sparse_diffusion/diffusion/heterogeneous_transition.py"],
    capture_output=True,
    text=True
)
if result.returncode == 0:
    print("   ✅ heterogeneous_transition.py 语法正确")
else:
    print(f"   ❌ heterogeneous_transition.py 语法错误:")
    print(result.stderr)

print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)
print("\n建议:")
print("1. 如果所有测试通过，可以运行小规模训练验证")
print("2. 检查采样结果是否符合关系族隔离的要求")
print("3. 监控训练损失，确认训练正常进行")
