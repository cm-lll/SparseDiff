"""
支持异质图关系族隔离的转移矩阵
"""
import torch
import sparse_diffusion.utils as utils


class HeterogeneousMarginalUniformTransition:
    """
    支持异质图关系族隔离的转移矩阵
    
    对于边，每个关系族有独立的转移矩阵，确保扩散时不会跨关系族。
    """
    def __init__(self, x_marginals, e_marginals, y_classes, charge_marginals,
                 edge_family_marginals=None, edge_family_offsets=None):
        """
        Args:
            x_marginals: 节点类型的边际分布
            e_marginals: 边类型的全局边际分布（用于同质图模式）
            y_classes: y 的类别数
            charge_marginals: charge 的边际分布
            edge_family_marginals: Dict[str, torch.Tensor] - 每个关系族的边类型边际分布
            edge_family_offsets: Dict[str, int] - 每个关系族的 offset
        """
        self.X_classes = len(x_marginals)
        self.E_classes = len(e_marginals)
        self.y_classes = y_classes
        self.x_marginals = x_marginals
        self.e_marginals = e_marginals  # 全局边类型分布（用于同质图或回退）
        self.charge_marginals = charge_marginals
        
        # 异质图相关
        self.edge_family_marginals = edge_family_marginals or {}
        self.edge_family_offsets = edge_family_offsets or {}
        self.heterogeneous = len(self.edge_family_marginals) > 0
        
        # 为每个关系族创建均匀分布矩阵
        if self.heterogeneous:
            self.edge_family_uniforms = {}
            for fam_name, fam_marginals in self.edge_family_marginals.items():
                # fam_marginals 包含 no-edge (0) + 子类别
                num_classes = len(fam_marginals)
                u_e = fam_marginals.unsqueeze(0).expand(num_classes, -1).unsqueeze(0)
                self.edge_family_uniforms[fam_name] = u_e
        else:
            self.edge_family_uniforms = {}
        
        # 全局均匀分布（用于同质图或回退）
        self.u_x = x_marginals.unsqueeze(0).expand(self.X_classes, -1).unsqueeze(0)
        self.u_e = e_marginals.unsqueeze(0).expand(self.E_classes, -1).unsqueeze(0)
        self.u_y = torch.ones(1, self.y_classes, self.y_classes)
        if self.y_classes > 0:
            self.u_y = self.u_y / self.y_classes
        
        if self.charge_marginals.numel() > 0:
            self.charge_classes = len(charge_marginals)
            self.u_charge = (
                charge_marginals.unsqueeze(0)
                .expand(self.charge_classes, -1)
                .unsqueeze(0)
            )
        else:
            self.charge_classes = 0
            self.u_charge = None

    def get_Qt(self, beta_t, device, edge_family_name=None):
        """
        返回单步转移矩阵
        
        Args:
            beta_t: (bs) 或 (bs, 1) 或标量，噪声水平
            device: 设备
            edge_family_name: 关系族名称（用于异质图模式）
        
        Returns:
            PlaceHolder(X=q_x, E=q_e, y=q_y, charge=q_charge)
        """
        # 处理不同维度的 beta_t
        beta_t = beta_t.to(device)
        if beta_t.dim() == 0:
            # 标量 -> (1, 1)
            beta_t = beta_t.unsqueeze(0).unsqueeze(0)
        elif beta_t.dim() == 1:
            # (bs,) -> (bs, 1)
            beta_t = beta_t.unsqueeze(1)
        elif beta_t.dim() == 2 and beta_t.shape[1] == 1:
            # (bs, 1) -> 保持不变
            pass
        else:
            # 其他情况，尝试 reshape
            beta_t = beta_t.view(-1, 1)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        eye_x = torch.eye(self.X_classes, device=device).unsqueeze(0)  # (1, X_classes, X_classes)
        # beta_t: (bs,) -> (bs, 1, 1) for broadcasting with (1, X_classes, X_classes)
        beta_t_expanded = beta_t.view(-1, 1, 1)
        q_x = beta_t_expanded * self.u_x + (1 - beta_t_expanded) * eye_x
        
        # 边的转移矩阵：根据是否指定关系族选择
        if self.heterogeneous and edge_family_name is not None:
            # 使用关系族特定的转移矩阵
            if edge_family_name in self.edge_family_uniforms:
                u_e_fam = self.edge_family_uniforms[edge_family_name].to(device)
                num_classes = u_e_fam.shape[1]
                eye_e = torch.eye(num_classes, device=device).unsqueeze(0)
                beta_t_expanded = beta_t.view(-1, 1, 1)
                q_e = beta_t_expanded * u_e_fam + (1 - beta_t_expanded) * eye_e
            else:
                # 回退到全局转移矩阵
                eye_e = torch.eye(self.E_classes, device=device).unsqueeze(0)
                beta_t_expanded = beta_t.view(-1, 1, 1)
                q_e = beta_t_expanded * self.u_e + (1 - beta_t_expanded) * eye_e
        else:
            # 同质图模式：使用全局转移矩阵
            eye_e = torch.eye(self.E_classes, device=device).unsqueeze(0)
            beta_t_expanded = beta_t.view(-1, 1, 1)
            q_e = beta_t_expanded * self.u_e + (1 - beta_t_expanded) * eye_e
        
        eye_y = torch.eye(self.y_classes, device=device).unsqueeze(0)
        beta_t_expanded = beta_t.view(-1, 1, 1)
        q_y = beta_t_expanded * self.u_y + (1 - beta_t_expanded) * eye_y

        q_charge = None
        if self.charge_marginals.numel() > 0:
            self.u_charge = self.u_charge.to(device)
            eye_charge = torch.eye(self.charge_classes, device=device).unsqueeze(0)
            beta_t_expanded = beta_t.view(-1, 1, 1)
            q_charge = beta_t_expanded * self.u_charge + (1 - beta_t_expanded) * eye_charge

        return utils.PlaceHolder(X=q_x, E=q_e, y=q_y, charge=q_charge)

    def get_Qt_bar(self, alpha_bar_t, device, edge_family_name=None):
        """
        返回 t 步累积转移矩阵
        
        Args:
            alpha_bar_t: (bs) 或 (bs, 1) 或标量，累积的 alpha_bar
            device: 设备
            edge_family_name: 关系族名称（用于异质图模式）
        
        Returns:
            PlaceHolder(X=q_x, E=q_e, y=q_y, charge=q_charge)
        """
        # 处理不同维度的 alpha_bar_t
        alpha_bar_t = alpha_bar_t.to(device)
        if alpha_bar_t.dim() == 0:
            # 标量 -> (1, 1)
            alpha_bar_t = alpha_bar_t.unsqueeze(0).unsqueeze(0)
        elif alpha_bar_t.dim() == 1:
            # (bs,) -> (bs, 1)
            alpha_bar_t = alpha_bar_t.unsqueeze(1)
        elif alpha_bar_t.dim() == 2 and alpha_bar_t.shape[1] == 1:
            # (bs, 1) -> 保持不变
            pass
        else:
            # 其他情况，尝试 reshape
            alpha_bar_t = alpha_bar_t.view(-1, 1)
        self.u_x = self.u_x.to(device)
        self.u_e = self.u_e.to(device)
        self.u_y = self.u_y.to(device)

        eye_x = torch.eye(self.X_classes, device=device).unsqueeze(0)  # (1, X_classes, X_classes)
        alpha_bar_t_expanded = alpha_bar_t.view(-1, 1, 1)
        q_x = (
            alpha_bar_t_expanded * eye_x
            + (1 - alpha_bar_t_expanded) * self.u_x
        )
        
        # 边的转移矩阵：根据是否指定关系族选择
        if self.heterogeneous and edge_family_name is not None:
            # 使用关系族特定的转移矩阵
            if edge_family_name in self.edge_family_uniforms:
                u_e_fam = self.edge_family_uniforms[edge_family_name].to(device)
                num_classes = u_e_fam.shape[1]
                eye_e = torch.eye(num_classes, device=device).unsqueeze(0)
                alpha_bar_t_expanded = alpha_bar_t.view(-1, 1, 1)
                q_e = (
                    alpha_bar_t_expanded * eye_e
                    + (1 - alpha_bar_t_expanded) * u_e_fam
                )
            else:
                # 回退到全局转移矩阵
                eye_e = torch.eye(self.E_classes, device=device).unsqueeze(0)
                alpha_bar_t_expanded = alpha_bar_t.view(-1, 1, 1)
                q_e = (
                    alpha_bar_t_expanded * eye_e
                    + (1 - alpha_bar_t_expanded) * self.u_e
                )
        else:
            # 同质图模式：使用全局转移矩阵
            eye_e = torch.eye(self.E_classes, device=device).unsqueeze(0)
            alpha_bar_t_expanded = alpha_bar_t.view(-1, 1, 1)
            q_e = (
                alpha_bar_t_expanded * eye_e
                + (1 - alpha_bar_t_expanded) * self.u_e
            )
        
        eye_y = torch.eye(self.y_classes, device=device).unsqueeze(0)
        alpha_bar_t_expanded = alpha_bar_t.view(-1, 1, 1)
        q_y = (
            alpha_bar_t_expanded * eye_y
            + (1 - alpha_bar_t_expanded) * self.u_y
        )

        q_charge = None
        if self.charge_marginals.numel() > 0:
            self.u_charge = self.u_charge.to(device)
            eye_charge = torch.eye(self.charge_classes, device=device).unsqueeze(0)
            alpha_bar_t_expanded = alpha_bar_t.view(-1, 1, 1)
            q_charge = (
                alpha_bar_t_expanded * eye_charge
                + (1 - alpha_bar_t_expanded) * self.u_charge
            )

        return utils.PlaceHolder(X=q_x, E=q_e, y=q_y, charge=q_charge)
    
    def get_all_family_Qt(self, beta_t, device):
        """
        为所有关系族返回单步转移矩阵（用于批处理）
        
        Args:
            beta_t: (bs) 噪声水平
            device: 设备
        
        Returns:
            Dict[str, PlaceHolder] - 每个关系族的转移矩阵
        """
        if not self.heterogeneous:
            # 同质图模式：返回全局转移矩阵
            q_all = self.get_Qt(beta_t, device)
            return {"global": q_all}
        
        # 异质图模式：为每个关系族创建转移矩阵
        family_qt = {}
        for fam_name in self.edge_family_marginals.keys():
            family_qt[fam_name] = self.get_Qt(beta_t, device, edge_family_name=fam_name)
        
        return family_qt
    
    def get_all_family_Qt_bar(self, alpha_bar_t, device):
        """
        为所有关系族返回转移矩阵（用于批处理）
        
        Returns:
            Dict[str, PlaceHolder] - 每个关系族的转移矩阵
        """
        if not self.heterogeneous:
            # 同质图模式：返回全局转移矩阵
            q_all = self.get_Qt_bar(alpha_bar_t, device)
            return {"global": q_all}
        
        # 异质图模式：为每个关系族创建转移矩阵
        family_qt = {}
        for fam_name in self.edge_family_marginals.keys():
            family_qt[fam_name] = self.get_Qt_bar(alpha_bar_t, device, edge_family_name=fam_name)
        
        return family_qt
