import torch
from torch.nn import functional as F
import numpy as np
import math

from sparse_diffusion.utils import PlaceHolder
from sparse_diffusion import utils
from sparse_diffusion.diffusion.sample_edges import sample_query_edges, sampled_condensed_indices_uniformly
from sparse_diffusion.diffusion.sample_edges_utils import condensed_to_matrix_index_batch


def sum_except_batch(x):
    return x.reshape(x.size(0), -1).sum(dim=-1)


def assert_correctly_masked(variable, node_mask):
    assert (
        variable * (1 - node_mask.long())
    ).detach().abs().max() < 1e-4, "Variables not masked properly."


def sample_gaussian(size):
    x = torch.randn(size)
    return x


def sample_gaussian_with_mask(size, node_mask):
    x = torch.randn(size)
    x = x.type_as(node_mask.float())
    x_masked = x * node_mask
    return x_masked


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1. This may help improve stability during
    sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)

    alphas_step = alphas2[1:] / alphas2[:-1]

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def cosine_beta_schedule_discrete(timesteps, s=0.008, skip=1):
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ."""
    # steps = timesteps + 2
    # x = np.linspace(0, steps, steps)
    # # skip_index = np.concatenate([np.array([0]),np.arange(1, timesteps+1, skip),np.array([steps-1])])
    # skip_index = np.concatenate([np.array([0]),np.arange(skip, timesteps+1, skip),np.array([steps-1])])
    # x = x[skip_index]
    steps = timesteps + 2
    num_steps = timesteps//skip + 2
    x = np.linspace(0, steps, num_steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    return betas.squeeze()


def custom_beta_schedule_discrete(timesteps, average_num_nodes=50, s=0.008, skip=1):
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ."""
    # steps = timesteps + 2
    # x = np.linspace(0, steps, steps)
    # # skip_index = np.concatenate([np.array([0]),np.arange(1, timesteps+1, skip),np.array([steps-1])])
    # skip_index = np.concatenate([np.array([0]),np.arange(skip, timesteps+1, skip),np.array([steps-1])])
    # x = x[skip_index]
    steps = timesteps + 2
    num_steps = timesteps//skip + 2
    x = np.linspace(0, steps, num_steps)

    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas

    assert timesteps >= 100

    p = 4 / 5  # 1 - 1 / num_edge_classes
    num_edges = average_num_nodes * (average_num_nodes - 1) / 2

    # First 100 steps: only a few updates per graph
    updates_per_graph = 1.2
    beta_first = updates_per_graph / (p * num_edges)

    betas[betas < beta_first] = beta_first
    return np.array(betas)


def gaussian_KL(q_mu, q_sigma):
    """Computes the KL distance between a normal distribution and the standard normal.
    Args:
        q_mu: Mean of distribution q.
        q_sigma: Standard deviation of distribution q.
        p_mu: Mean of distribution p.
        p_sigma: Standard deviation of distribution p.
    Returns:
        The KL distance, summed over all dimensions except the batch dim.
    """
    return sum_except_batch(
        (torch.log(1 / q_sigma) + 0.5 * (q_sigma**2 + q_mu**2) - 0.5)
    )


def cdf_std_gaussian(x):
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))


def SNR(gamma):
    """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
    return torch.exp(-gamma)


def inflate_batch_array(array, target_shape):
    """
    Inflates the batch array (array) with only a single axis (i.e. shape = (batch_size,), or possibly more empty
    axes (i.e. shape (batch_size, 1, ..., 1)) to match the target shape.
    """
    target_shape = (array.size(0),) + (1,) * (len(target_shape) - 1)
    return array.view(target_shape)


def sigma(gamma, target_shape):
    """Computes sigma given gamma."""
    return inflate_batch_array(torch.sqrt(torch.sigmoid(gamma)), target_shape)


def alpha(gamma, target_shape):
    """Computes alpha given gamma."""
    return inflate_batch_array(torch.sqrt(torch.sigmoid(-gamma)), target_shape)


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def check_tensor_same_size(*args):
    for i, arg in enumerate(args):
        if i == 0:
            continue
        assert args[0].size() == arg.size()


def sigma_and_alpha_t_given_s(
    gamma_t: torch.Tensor, gamma_s: torch.Tensor, target_size: torch.Size
):
    """
    Computes sigma t given s, using gamma_t and gamma_s. Used during sampling.

    These are defined as:
        alpha t given s = alpha t / alpha s,
        sigma t given s = sqrt(1 - (alpha t given s) ^2 ).
    """
    sigma2_t_given_s = inflate_batch_array(
        -torch.expm1(F.softplus(gamma_s) - F.softplus(gamma_t)), target_size
    )

    # alpha_t_given_s = alpha_t / alpha_s
    log_alpha2_t = F.logsigmoid(-gamma_t)
    log_alpha2_s = F.logsigmoid(-gamma_s)
    log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

    alpha_t_given_s = torch.exp(0.5 * log_alpha2_t_given_s)
    alpha_t_given_s = inflate_batch_array(alpha_t_given_s, target_size)

    sigma_t_given_s = torch.sqrt(sigma2_t_given_s)

    return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s


def reverse_tensor(x):
    return x[torch.arange(x.size(0) - 1, -1, -1)]


def sample_discrete_features(probX, probE, node_mask, prob_charge=None):
    """Sample features from multinomial distribution with given probabilities (probX, probE, proby)
    :param probX: bs, n, dx_out        node features
    :param probE: bs, n, n, de_out     edge features
    :param proby: bs, dy_out           global features.
    """
    bs, n, _ = probX.shape
    # Noise X
    # The masked rows should define probability distributions as well
    probX[~node_mask] = 1 / probX.shape[-1]

    # Flatten the probability tensor to sample with multinomial
    probX = probX.reshape(bs * n, -1)  # (bs * n, dx_out)

    # Sample X
    X_t = probX.multinomial(1)  # (bs * n, 1)
    X_t = X_t.reshape(bs, n)  # (bs, n)

    # Noise E
    # The masked rows should define probability distributions as well
    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]

    probE = probE.reshape(bs * n * n, -1)  # (bs * n * n, de_out)

    # Sample E
    E_t = probE.multinomial(1).reshape(bs, n, n)  # (bs, n, n)
    E_t = torch.triu(E_t, diagonal=1)
    E_t = E_t + torch.transpose(E_t, 1, 2)

    charge_t = X_t.new_zeros((*X_t.shape[:-1], 0))
    if prob_charge is not None:
        prob_charge[~node_mask] = 1 / prob_charge.shape[-1]
        prob_charge = prob_charge.reshape(bs * n, -1)
        charge_t = prob_charge.multinomial(1)
        charge_t = charge_t.reshape(bs, n)

    return PlaceHolder(X=X_t, E=E_t, y=torch.zeros(bs, 0).type_as(X_t), charge=charge_t)


def sample_discrete_edge_features(probE, node_mask):
    """Sample features from multinomial distribution with given probabilities (probX, probE, proby)
    :param probX: bs, n, dx_out        node features
    :param probE: bs, n, n, de_out     edge features
    :param proby: bs, dy_out           global features.
    """
    bs, n, _, _ = probE.shape
    # Noise E
    # The masked rows should define probability distributions as well
    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]

    probE = probE.reshape(bs * n * n, -1)  # (bs * n * n, de_out)

    # Sample E
    E_t = probE.multinomial(1).reshape(bs, n, n)  # (bs, n, n)
    E_t = torch.triu(E_t, diagonal=1)
    E_t = E_t + torch.transpose(E_t, 1, 2)

    return E_t


def sample_discrete_node_features(probX, node_mask):
    """Sample features from multinomial distribution with given probabilities (probX, probE, proby)
    :param probX: bs, n, dx_out        node features
    :param probE: bs, n, n, de_out     edge features
    :param proby: bs, dy_out           global features.
    """
    bs, n, _ = probX.shape
    # Noise X
    # The masked rows should define probability distributions as well
    probX[~node_mask] = 1 / probX.shape[-1]

    # Flatten the probability tensor to sample with multinomial
    probX = probX.reshape(bs * n, -1)  # (bs * n, dx_out)

    # Sample X
    X_t = probX.multinomial(1)  # (bs * n, 1)
    X_t = X_t.reshape(bs, n)  # (bs, n)

    return X_t


def compute_posterior_distribution(M, M_t, Qt_M, Qsb_M, Qtb_M):
    """M: X or E
    Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T
    """
    # Flatten feature tensors
    M = M.flatten(start_dim=1, end_dim=-2).to(
        torch.float32
    )  # (bs, N, d) with N = n or n * n
    M_t = M_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)  # same

    Qt_M_T = torch.transpose(Qt_M, -2, -1)  # (bs, d, d)

    left_term = M_t @ Qt_M_T  # (bs, N, d)
    right_term = M @ Qsb_M  # (bs, N, d)
    product = left_term * right_term  # (bs, N, d)

    denom = M @ Qtb_M  # (bs, N, d) @ (bs, d, d) = (bs, N, d)
    denom = (denom * M_t).sum(dim=-1)  # (bs, N, d) * (bs, N, d) + sum = (bs, N)

    # mask out where denom is 0.
    denom[denom == 0.] = 1

    prob = product / denom.unsqueeze(-1)  # (bs, N, d)

    return prob


def compute_sparse_posterior_distribution(M, M_t, Qt_M, Qsb_M, Qtb_M):
    """M: node or edge_attr: n * dx (or m * de)
    Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T
    """
    # Flatten feature tensors
    Qt_M_T = torch.transpose(Qt_M, -2, -1)  # (bs, d, d)

    left_term = M_t.unsqueeze(1) @ Qt_M_T  # (n, 1, d) @ (n, d, d) = (n, 1, d)
    right_term = M.unsqueeze(1) @ Qsb_M  # (n, 1, d) @ (n, d, d) = (n, 1, d)
    product = left_term.squeeze(1) * right_term.squeeze(1)  # (n, d)

    denom = M.unsqueeze(1) @ Qtb_M  # (n, 1, d) @ (n, d, d) = (n, 1, d)
    denom = (denom.squeeze(1) * M_t).sum(dim=-1)  # (n, d) * (n, d) + sum = (n)

    prob = product / denom.unsqueeze(-1)  # (n, d)

    return prob


def compute_batched_over0_posterior_distribution(X_t, Qt, Qsb, Qtb):
    """M: X or E
    Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0
    X_t: bs, n, dt          or bs, n, n, dt
    Qt: bs, d_t-1, dt
    Qsb: bs, d0, d_t-1
    Qtb: bs, d0, dt.
    """
    # Flatten feature tensors
    # Careful with this line. It does nothing if X is a node feature. If X is an edge features it maps to
    # bs x (n ** 2) x d
    X_t = X_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)  # bs x N x dt

    Qt_T = Qt.transpose(-1, -2)  # bs, dt, d_t-1
    left_term = X_t @ Qt_T  # bs, N, d_t-1
    left_term = left_term.unsqueeze(dim=2)  # bs, N, 1, d_t-1

    right_term = Qsb.unsqueeze(1)  # bs, 1, d0, d_t-1
    numerator = left_term * right_term  # bs, N, d0, d_t-1

    X_t_transposed = X_t.transpose(-1, -2)  # bs, dt, N

    prod = Qtb @ X_t_transposed  # bs, d0, N
    prod = prod.transpose(-1, -2)  # bs, N, d0
    denominator = prod.unsqueeze(-1)  # bs, N, d0, 1
    denominator[denominator == 0] = 1e-6

    out = numerator / denominator
    return out


def mask_distributions(
    true_X, true_E, pred_X, pred_E, node_mask, true_charge=None, pred_charge=None
):
    # Set masked rows to arbitrary distributions, so it doesn't contribute to loss
    row_X = torch.zeros(true_X.size(-1), dtype=torch.float, device=true_X.device)
    row_X[0] = 1.0
    row_E = torch.zeros(true_E.size(-1), dtype=torch.float, device=true_E.device)
    row_E[0] = 1.0

    diag_mask = ~torch.eye(
        node_mask.size(1), device=node_mask.device, dtype=torch.bool
    ).unsqueeze(0)
    true_X[~node_mask] = row_X
    pred_X[~node_mask] = row_X
    true_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E
    pred_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E

    # Add a small value everywhere to avoid nans
    pred_X = pred_X + 1e-7
    pred_E = pred_E + 1e-7
    pred_X = pred_X / torch.sum(pred_X, dim=-1, keepdim=True)
    pred_E = pred_E / torch.sum(pred_E, dim=-1, keepdim=True)

    if true_charge is not None and pred_charge is not None:
        row_charge = torch.zeros(
            true_charge.size(-1), dtype=torch.float, device=true_charge.device
        )
        row_charge[0] = 1.0
        true_charge[~node_mask] = row_charge
        pred_charge[~node_mask] = row_charge

        pred_charge = pred_charge + 1e-7
        pred_charge = pred_charge / torch.sum(pred_charge, dim=-1, keepdim=True)

    return true_X, true_E, pred_X, pred_E, true_charge, pred_charge


def posterior_distributions(X, E, X_t, E_t, y_t, Qt, Qsb, Qtb, charge, charge_t):
    prob_X = compute_posterior_distribution(
        M=X, M_t=X_t, Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X
    )  # (bs, n, dx)
    prob_E = compute_posterior_distribution(
        M=E, M_t=E_t, Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E
    )  # (bs, n * n, de)

    prob_charge = None
    if charge is not None and charge_t is not None:
        prob_charge = compute_posterior_distribution(
            M=charge, M_t=charge_t, Qt_M=Qt.charge, Qsb_M=Qsb.charge, Qtb_M=Qtb.charge
        )

    return PlaceHolder(X=prob_X, E=prob_E, y=y_t, charge=prob_charge)


def posterior_distributions_heterogeneous(
    X, E, E_t, X_t, y_t, Qt_dict, Qsb_dict, Qtb_dict, 
    edge_family_offsets, num_global_states, charge=None, charge_t=None, Qt_charge=None, Qsb_charge=None, Qtb_charge=None
):
    """
    为异质图计算后验分布，为每个关系族使用独立的转移矩阵
    
    Args:
        X: (bs, n, dx) 节点特征
        E: (bs, n, n, de) 边特征（全局状态空间）
        E_t: (bs, n, n, de) 噪声边特征
        X_t: (bs, n, dx) 噪声节点特征
        y_t: (bs, dy) 图级别特征
        Qt_dict: Dict[str, PlaceHolder] - 每个关系族的单步转移矩阵
        Qsb_dict: Dict[str, PlaceHolder] - 每个关系族的累积转移矩阵（s步）
        Qtb_dict: Dict[str, PlaceHolder] - 每个关系族的累积转移矩阵（t步）
        edge_family_offsets: Dict[str, int] - 每个关系族的全局ID偏移
        num_global_states: int - 全局状态空间大小
        charge: (bs, n, d_charge) 电荷特征
        charge_t: (bs, n, d_charge) 噪声电荷特征
        Qt_charge, Qsb_charge, Qtb_charge: 电荷的转移矩阵
    
    Returns:
        PlaceHolder(X=prob_X, E=prob_E, y=y_t, charge=prob_charge)
    """
    # 节点和电荷的后验分布（使用全局转移矩阵，因为它们不受关系族影响）
    # 使用第一个关系族的转移矩阵的X部分（所有关系族共享节点转移矩阵）
    first_fam_name = list(Qt_dict.keys())[0]
    Qt_X = Qt_dict[first_fam_name].X
    Qsb_X = Qsb_dict[first_fam_name].X
    Qtb_X = Qtb_dict[first_fam_name].X
    
    prob_X = compute_posterior_distribution(
        M=X, M_t=X_t, Qt_M=Qt_X, Qsb_M=Qsb_X, Qtb_M=Qtb_X
    )  # (bs, n, dx)
    
    prob_charge = None
    if charge is not None and charge_t is not None and Qt_charge is not None:
        prob_charge = compute_posterior_distribution(
            M=charge, M_t=charge_t, Qt_M=Qt_charge, Qsb_M=Qsb_charge, Qtb_M=Qtb_charge
        )
    
    # 边的后验分布：为每个关系族独立计算
    bs, n, n, de = E.shape
    E_flat = E.reshape(bs, n * n, de)  # (bs, n*n, de)
    E_t_flat = E_t.reshape(bs, n * n, de)  # (bs, n*n, de)
    
    # 根据 E_t 的全局ID推断每个边属于哪个关系族
    E_t_discrete = E_t_flat.argmax(dim=-1)  # (bs, n*n) - 全局ID
    
    # 初始化全局状态空间的概率
    prob_E_flat = torch.zeros_like(E_flat)  # (bs, n*n, de)
    
    # 为每个关系族计算后验分布
    for fam_name, offset in edge_family_offsets.items():
        if fam_name not in Qt_dict:
            continue
        
        # 找到下一个关系族的offset
        next_offset = num_global_states
        for other_fam_name, other_offset in edge_family_offsets.items():
            if other_offset > offset and other_offset < next_offset:
                next_offset = other_offset
        
        # 判断哪些边属于这个关系族
        # 边属于该关系族如果：E_t == 0 (no-edge) 或 E_t 在 [offset, next_offset) 范围内
        fam_mask = (E_t_discrete == 0) | ((E_t_discrete >= offset) & (E_t_discrete < next_offset))  # (bs, n*n)
        
        if not fam_mask.any():
            continue
        
        # 获取该关系族的边
        E_fam = E_flat[fam_mask]  # (num_edges_fam, de)
        E_t_fam = E_t_flat[fam_mask]  # (num_edges_fam, de)
        
        # 获取该关系族的转移矩阵
        Qt_fam = Qt_dict[fam_name].E  # (bs, num_fam_states, num_fam_states)
        Qsb_fam = Qsb_dict[fam_name].E  # (bs, num_fam_states, num_fam_states)
        Qtb_fam = Qtb_dict[fam_name].E  # (bs, num_fam_states, num_fam_states)
        
        # 将全局状态转换为局部状态
        E_t_fam_discrete = E_t_fam.argmax(dim=-1)  # (num_edges_fam,)
        E_t_fam_local = E_t_fam_discrete.clone()
        non_zero_mask = E_t_fam_local != 0
        if non_zero_mask.any():
            E_t_fam_local[non_zero_mask] = E_t_fam_local[non_zero_mask] - offset + 1
        
        # 转换为局部状态的one-hot编码
        num_fam_states = Qt_fam.shape[-1]
        E_t_fam_local_onehot = torch.nn.functional.one_hot(
            E_t_fam_local.long(), num_classes=num_fam_states
        ).float()  # (num_edges_fam, num_fam_states)
        
        # 同样处理 E_fam
        E_fam_discrete = E_fam.argmax(dim=-1)  # (num_edges_fam,)
        E_fam_local = E_fam_discrete.clone()
        non_zero_mask = E_fam_local != 0
        if non_zero_mask.any():
            E_fam_local[non_zero_mask] = E_fam_local[non_zero_mask] - offset + 1
        E_fam_local_onehot = torch.nn.functional.one_hot(
            E_fam_local.long(), num_classes=num_fam_states
        ).float()  # (num_edges_fam, num_fam_states)
        
        # 获取每个边所属的batch
        batch_indices = torch.arange(bs, device=E.device).repeat_interleave(
            (fam_mask.sum(dim=1))  # 每个batch中属于该关系族的边数
        )[:fam_mask.sum()]
        
        # 为每个边选择对应的batch转移矩阵
        Qt_fam_batch = Qt_fam[batch_indices]  # (num_edges_fam, num_fam_states, num_fam_states)
        Qsb_fam_batch = Qsb_fam[batch_indices]  # (num_edges_fam, num_fam_states, num_fam_states)
        Qtb_fam_batch = Qtb_fam[batch_indices]  # (num_edges_fam, num_fam_states, num_fam_states)
        
        # 计算局部状态空间的后验分布
        prob_E_fam_local = compute_posterior_distribution(
            M=E_fam_local_onehot.unsqueeze(0),  # (1, num_edges_fam, num_fam_states)
            M_t=E_t_fam_local_onehot.unsqueeze(0),  # (1, num_edges_fam, num_fam_states)
            Qt_M=Qt_fam_batch.unsqueeze(0),  # (1, num_edges_fam, num_fam_states, num_fam_states)
            Qsb_M=Qsb_fam_batch.unsqueeze(0),  # (1, num_edges_fam, num_fam_states, num_fam_states)
            Qtb_M=Qtb_fam_batch.unsqueeze(0),  # (1, num_edges_fam, num_fam_states, num_fam_states)
        )  # (1, num_edges_fam, num_fam_states)
        prob_E_fam_local = prob_E_fam_local.squeeze(0)  # (num_edges_fam, num_fam_states)
        
        # 映射回全局状态空间
        prob_E_fam_global = torch.zeros(
            (prob_E_fam_local.shape[0], num_global_states),
            device=E.device, dtype=prob_E_fam_local.dtype
        )
        for local_state in range(num_fam_states):
            if local_state == 0:
                global_state = 0
            else:
                global_state = offset + local_state - 1
            if global_state < num_global_states:
                prob_E_fam_global[:, global_state] = prob_E_fam_local[:, local_state]
        
        # 填充到全局概率矩阵
        prob_E_flat[fam_mask] = prob_E_fam_global
    
    prob_E = prob_E_flat.reshape(bs, n, n, de)
    
    return PlaceHolder(X=prob_X, E=prob_E, y=y_t, charge=prob_charge)


def sample_discrete_feature_noise(limit_dist, node_mask):
    """Sample from the limit distribution of the diffusion process"""

    bs, n_max = node_mask.shape
    x_limit = limit_dist.X[None, None, :].expand(bs, n_max, -1)
    e_limit = limit_dist.E[None, None, None, :].expand(bs, n_max, n_max, -1)
    U_X = x_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max)
    U_E = e_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max, n_max)
    U_y = torch.empty((bs, 0))

    long_mask = node_mask.long()
    U_X = U_X.type_as(long_mask)
    U_E = U_E.type_as(long_mask)
    U_y = U_y.type_as(long_mask)

    U_X = F.one_hot(U_X, num_classes=x_limit.shape[-1]).float()
    U_E = F.one_hot(U_E, num_classes=e_limit.shape[-1]).float()

    # Get upper triangular part of edge noise, without main diagonal
    upper_triangular_mask = torch.zeros_like(U_E)
    indices = torch.triu_indices(row=U_E.size(1), col=U_E.size(2), offset=1)
    upper_triangular_mask[:, indices[0], indices[1], :] = 1

    U_E = U_E * upper_triangular_mask
    U_E = U_E + torch.transpose(U_E, 1, 2)

    assert (U_E == torch.transpose(U_E, 1, 2)).all()

    return PlaceHolder(X=U_X, E=U_E, y=U_y).mask(node_mask)


def sample_sparse_discrete_feature_noise(limit_dist, node_mask):
    """Sample from the limit distribution of the diffusion process"""
    # params
    bs, n_max = node_mask.shape
    device = node_mask.device
    batch = torch.where(node_mask > 0)[0]

    # get number of nodes and existnig edges
    n_node = node_mask.sum().int()  # (1, )
    n_nodes = node_mask.sum(-1)  # (bs, )
    n_edges = (n_nodes - 1) * n_nodes / 2  # (bs, )
    n_exist_edges = (
        torch.distributions.binomial.Binomial(n_edges, limit_dist.E[1:].sum())
        .sample()
        .long()
        .to(device)
    )  # (bs, )

    # expand dimensions
    x_limit = limit_dist.X[None, :].expand(n_node, -1)  # (n_node, dx)
    e_limit = limit_dist.E[None, :].expand(n_exist_edges.sum(), -1)  # (n_edge, de)

    # sample nodes and existing edges
    node = x_limit.multinomial(1)[:, 0]
    edge_attr = e_limit[:, 1:].multinomial(1)[:, 0] + 1
    node = F.one_hot(node, num_classes=x_limit.shape[-1]).float()
    edge_attr = F.one_hot(edge_attr, num_classes=e_limit.shape[-1]).float()
    y = torch.empty((bs, 0)).long()

    # sample edge index
    edge_index, _ = sample_query_edges(
        num_nodes_per_graph=n_nodes,
        edge_proportion=None,
        num_edges_to_sample=n_exist_edges,
    )

    # Get upper triangular part of edge noise, without main diagonal
    edge_index, edge_attr = utils.to_undirected(edge_index, edge_attr)

    # Sample charge
    charge = node.new_zeros((*node.shape[:-1], 0))
    if limit_dist.charge.numel() > 0:
        charge_limit = limit_dist.charge[None, :].expand(n_node, -1)
        charge = charge_limit.multinomial(1)[:, 0]
        charge = F.one_hot(charge, num_classes=charge_limit.shape[-1]).float()

    ptr = torch.unique(batch, sorted=True, return_counts=True)[1]
    ptr = torch.hstack([torch.tensor([0]).to(device), ptr.cumsum(-1)]).long()

    return utils.SparsePlaceHolder(
        node=node, edge_index=edge_index.long(), edge_attr=edge_attr, y=y, charge=charge,
        batch=batch, ptr=ptr
    ).to_device(device)


def sample_sparse_discrete_feature_noise_heterogeneous(limit_dist, node_mask, dataset_info, out_dims_E, device):
    """异质图：从 limit 采节点；边数按训练集每族平均 edge_family_avg_edge_counts 初始化，
    边对在各自关系族的合法 (src,dst) 上均匀采样，边类型从 limit_dist.E 的该族 slice 采样。
    这样 z_T 的边数接近「真实图平均」，便于 |Eq|=k*m 的 m 在第一步就有合理起点。
    """
    node_mask = node_mask.to(device)
    bs, n_max = node_mask.shape
    batch = torch.where(node_mask > 0)[0]  # (N,) 每个节点所属图
    n_node = node_mask.sum().int().item()
    n_nodes = node_mask.sum(-1).long()  # (bs,)

    # 获取必要的统计信息
    edge_family_avg_edge_counts = getattr(dataset_info, "edge_family_avg_edge_counts", {}) or {}
    edge_family_offsets = getattr(dataset_info, "edge_family_offsets", {}) or {}
    fam_endpoints = getattr(dataset_info, "fam_endpoints", {}) or {}
    type_offsets = getattr(dataset_info, "type_offsets", {}) or {}
    id2edge_family = {v: k for k, v in getattr(dataset_info, "edge_family2id", {}).items()}
    
    if not edge_family_avg_edge_counts or not fam_endpoints or not type_offsets:
        # 缺少异质信息时退回通用初始化
        return sample_sparse_discrete_feature_noise(limit_dist, node_mask)
    
    # 节点初始化：按照用户描述的逻辑
    # 1. 先按节点类型分布确定每个类型的节点数量
    # 2. 然后对每个类型，按照该类型内的子类别分布分子类别
    node_type_distribution = getattr(dataset_info, "node_type_distribution", None)
    node_subtype_by_type = getattr(dataset_info, "node_subtype_by_type", None)
    
    # 获取节点类型名称列表（从type_offsets的键获取）
    node_type_names = list(type_offsets.keys()) if type_offsets else []
    
    # type_sizes（节点类型子类个数），与 sample_p_zs_given_zt 一致
    total_node_subtypes = int(limit_dist.X.shape[-1])
    type_sizes = {}
    sorted_ty = sorted(type_offsets.items(), key=lambda x: x[1])
    for i, (t, off) in enumerate(sorted_ty):
        if i + 1 < len(sorted_ty):
            type_sizes[t] = sorted_ty[i + 1][1] - off
        else:
            # 最后一个类型的大小由全局子类别空间上界确定，避免临时值带来越界风险
            type_sizes[t] = max(1, total_node_subtypes - off)
    
    x_limit = limit_dist.X[None, :].expand(n_node, -1).to(device) if n_node > 0 else None
    
    if node_type_distribution and node_subtype_by_type and len(node_type_names) > 0:
        # 使用新的逻辑：先按类型分布，再按子类别分布
        node_t = torch.zeros(n_node, dtype=torch.long, device=device)
        node = torch.zeros(n_node, x_limit.shape[-1], dtype=torch.float, device=device)
        
        # 为每个图分别处理
        for b in range(bs):
            batch_mask = (batch == b)
            n_b = batch_mask.sum().item()
            if n_b == 0:
                continue
            
            # 1. 按节点类型分布确定每个类型的节点数量
            node_type_counts = {}
            remaining = n_b
            for i, node_type_name in enumerate(node_type_names):
                if i == len(node_type_names) - 1:
                    # 最后一个类型，分配剩余的所有节点
                    node_type_counts[node_type_name] = remaining
                else:
                    prob = node_type_distribution.get(node_type_name, 1.0 / len(node_type_names))
                    count = int(round(prob * n_b))
                    count = min(count, remaining)
                    node_type_counts[node_type_name] = count
                    remaining -= count
            
            # 2. 对每个类型，按照该类型内的子类别分布分子类别
            node_idx_in_batch = torch.where(batch_mask)[0]
            current_idx = 0
            for node_type_name, count in node_type_counts.items():
                if count == 0:
                    continue
                if node_type_name not in type_offsets or node_type_name not in type_sizes:
                    continue
                
                offset = type_offsets[node_type_name]
                type_size = type_sizes[node_type_name]
                
                # 获取该类型的子类别分布
                subtype_dist = node_subtype_by_type.get(node_type_name)
                if subtype_dist is None:
                    # 如果没有子类别分布，使用均匀分布
                    subtype_dist = torch.ones(type_size, dtype=torch.float) / type_size
                else:
                    # 确保是tensor且归一化
                    if isinstance(subtype_dist, torch.Tensor):
                        subtype_dist = subtype_dist.to(device)
                    else:
                        subtype_dist = torch.tensor(subtype_dist, dtype=torch.float, device=device)
                    if subtype_dist.sum() > 0:
                        subtype_dist = subtype_dist / subtype_dist.sum()
                    else:
                        subtype_dist = torch.ones(type_size, dtype=torch.float, device=device) / type_size
                
                # 从子类别分布中采样
                if count > 0:
                    selected_indices = node_idx_in_batch[current_idx:current_idx + count]
                    subtype_samples = subtype_dist.multinomial(count, replacement=True)  # (count,)
                    global_subtype_ids = offset + subtype_samples  # (count,)
                    
                    # 设置节点类型和子类别
                    node_t[selected_indices] = global_subtype_ids
                    node[selected_indices] = F.one_hot(global_subtype_ids, num_classes=x_limit.shape[-1]).float()
                    current_idx += count
        
        # 更新type_sizes（使用实际的node_t最大值）
        if node_t.numel() > 0:
            for i, (t, off) in enumerate(sorted_ty):
                if i + 1 < len(sorted_ty):
                    type_sizes[t] = sorted_ty[i + 1][1] - off
                else:
                    type_sizes[t] = max(1, total_node_subtypes - off)
    else:
        # 回退到原来的逻辑：从全局分布采样
        node = x_limit.multinomial(1)[:, 0]
        node = F.one_hot(node, num_classes=x_limit.shape[-1]).float()
        node_t = node.argmax(dim=-1)  # (N,)
        
        # 更新type_sizes
        for i, (t, off) in enumerate(sorted_ty):
            if i + 1 < len(sorted_ty):
                type_sizes[t] = sorted_ty[i + 1][1] - off
            else:
                type_sizes[t] = max(1, total_node_subtypes - off)
    
    y = torch.empty((bs, 0)).long().to(device)
    charge = node.new_zeros((*node.shape[:-1], 0))
    if limit_dist.charge.numel() > 0:
        charge_limit = limit_dist.charge[None, :].expand(n_node, -1).to(device)
        charge = charge_limit.multinomial(1)[:, 0]
        charge = F.one_hot(charge, num_classes=charge_limit.shape[-1]).float()

    ptr = torch.unique(batch, sorted=True, return_counts=True)[1]
    ptr = torch.hstack([torch.tensor([0], device=device, dtype=torch.long), ptr.cumsum(-1)]).long()

    E_limit = limit_dist.E.float().to(device)
    all_ei, all_ea = [], []
    
    # 获取关系族分布和边子类别分布
    edge_family_distribution = getattr(dataset_info, "edge_family_distribution", None)
    edge_subtype_by_family = getattr(dataset_info, "edge_subtype_by_family", None)
    edge_family_marginals = getattr(dataset_info, "edge_family_marginals", None)
    edge_family2id = getattr(dataset_info, "edge_family2id", {}) or {}

    # 边初始化：按照用户描述的逻辑
    # 1. 先确定两个节点类型之间可能的关系族
    # 2. 按照关系族分布确定各个关系族的边数量
    # 3. 然后在对应的子类别中均匀选取边
    
    for b in range(bs):
        batch_mask = (batch == b)
        batch_node_t = node_t[batch_mask]  # 当前图的节点类型（全局子类别ID）
        batch_nodes_global = torch.where(batch_mask)[0]
        if batch_nodes_global.numel() == 0:
            continue
        # 固定当前图的哈希基数，确保跨类型对/跨关系族去重使用同一编码空间
        graph_hash_base = int(batch_nodes_global.max().item()) + 1
        # 记录当前图已经被任意关系族占用的有向边，保证同一 (u, v) 只保留一种关系
        used_edge_hash = set()
        
        # 找到所有可能的节点类型对（src_type, dst_type）
        type_pairs = set()
        for fam_name, endpoints in fam_endpoints.items():
            src_type = endpoints.get("src_type")
            dst_type = endpoints.get("dst_type")
            if src_type and dst_type and src_type in type_offsets and dst_type in type_offsets:
                type_pairs.add((src_type, dst_type))
        
        # 对每个类型对，按照关系族分布确定边数量
        for src_type, dst_type in type_pairs:
            src_offset = type_offsets[src_type]
            dst_offset = type_offsets[dst_type]
            src_size = type_sizes.get(src_type, 0)
            dst_size = type_sizes.get(dst_type, 0)
            if src_size <= 0 or dst_size <= 0:
                continue
            
            src_mask = (batch_node_t >= src_offset) & (batch_node_t < src_offset + src_size)
            dst_mask = (batch_node_t >= dst_offset) & (batch_node_t < dst_offset + dst_size)
            batch_src = torch.where(batch_mask)[0][src_mask]
            batch_dst = torch.where(batch_mask)[0][dst_mask]
            
            if len(batch_src) == 0 or len(batch_dst) == 0:
                continue
            
            num_src, num_dst = len(batch_src), len(batch_dst)
            if src_type == dst_type:
                num_possible = num_src * num_dst - num_src
            else:
                num_possible = num_src * num_dst
            if num_possible <= 0:
                continue
            
            # 获取该类型对的关系族分布
            type_pair = (src_type, dst_type)
            fam_dist = edge_family_distribution.get(type_pair, {}) if edge_family_distribution else {}
            
            # 如果没有关系族分布，使用edge_family_avg_edge_counts的旧逻辑
            if not fam_dist:
                # 回退到旧逻辑：直接使用edge_family_avg_edge_counts
                for fam_id, fam_name in id2edge_family.items():
                    if fam_name not in fam_endpoints:
                        continue
                    ep = fam_endpoints[fam_name]
                    if ep.get("src_type") != src_type or ep.get("dst_type") != dst_type:
                        continue
                    
                    m_fam = edge_family_avg_edge_counts.get(fam_name, 0.0)
                    n_exist_fam = max(0, int(round(m_fam)))
                    if n_exist_fam <= 0:
                        continue
                    
                    n_fam_b = min(n_exist_fam, num_possible)
                    if n_fam_b <= 0:
                        continue
                    
                    # 采样边对和边类型（使用旧逻辑）
                    n_added = _add_edges_for_family(
                        batch_src, batch_dst, src_type, dst_type, num_src, num_dst,
                        fam_name, fam_id, n_fam_b, num_possible,
                        edge_family_offsets, edge_subtype_by_family, edge_family_marginals,
                        E_limit, out_dims_E, device, all_ei, all_ea, edge_family2id,
                        forbidden_hash=used_edge_hash, hash_base=graph_hash_base,
                    )
                    num_possible = max(num_possible - n_added, 0)
            else:
                # 新逻辑：按照关系族分布确定各个关系族的边数量
                total_edges_for_pair = min(int(round(sum(edge_family_avg_edge_counts.get(fam_name, 0.0) 
                                                         for fam_name in fam_dist.keys()))), num_possible)
                
                remaining_edges = total_edges_for_pair
                for fam_name, fam_prob in fam_dist.items():
                    if remaining_edges <= 0:
                        break
                    
                    if fam_name not in edge_family2id:
                        continue
                    fam_id = edge_family2id[fam_name]
                    
                    # 计算该关系族的边数量
                    n_fam = int(round(fam_prob * total_edges_for_pair))
                    n_fam = min(n_fam, remaining_edges, num_possible)
                    if n_fam <= 0:
                        continue
                    
                    # 采样边对和边类型
                    n_added = _add_edges_for_family(
                        batch_src, batch_dst, src_type, dst_type, num_src, num_dst,
                        fam_name, fam_id, n_fam, num_possible,
                        edge_family_offsets, edge_subtype_by_family, edge_family_marginals,
                        E_limit, out_dims_E, device, all_ei, all_ea, edge_family2id,
                        forbidden_hash=used_edge_hash, hash_base=graph_hash_base,
                    )
                    
                    remaining_edges = max(remaining_edges - n_added, 0)
                    num_possible = max(num_possible - n_added, 0)  # 更新剩余可能的边数

    if len(all_ei) == 0:
        edge_index = torch.zeros(2, 0, device=device, dtype=torch.long)
        edge_attr = F.one_hot(torch.zeros(0, device=device, dtype=torch.long), num_classes=out_dims_E).float()
    else:
        edge_index = torch.cat(all_ei, dim=1)
        edge_attr_d = torch.cat(all_ea, dim=0)
        # 异质图：关系是有向的，不应该转换为无向边
        # 否则会导致反向边（如Paper->Author）没有匹配的关系族，只能允许no-edge
        # 例如：author_of是Author->Paper，不应该添加Paper->Author
        # 因此异质图模式下不使用to_undirected，保持有向边
        edge_attr = F.one_hot(edge_attr_d.long(), num_classes=out_dims_E).float()

    return utils.SparsePlaceHolder(
        node=node, edge_index=edge_index, edge_attr=edge_attr, y=y, charge=charge,
        batch=batch, ptr=ptr
    ).to_device(device)


def _add_edges_for_family(batch_src, batch_dst, src_type, dst_type, num_src, num_dst,
                         fam_name, fam_id, n_fam, num_possible,
                         edge_family_offsets, edge_subtype_by_family, edge_family_marginals, E_limit,
                         out_dims_E, device, all_ei, all_ea, edge_family2id=None,
                         forbidden_hash=None, hash_base=None):
    """辅助函数：为特定关系族添加边"""
    if n_fam <= 0:
        return 0
    if forbidden_hash is None:
        forbidden_hash = set()

    offset = edge_family_offsets.get(fam_name, 0)
    next_offset = out_dims_E
    for _, o in edge_family_offsets.items():
        if o > offset and o < next_offset:
            next_offset = o
    
    if src_type == dst_type:
        # Directed same-type relations use n*(n-1) ordered pairs (exclude self-loops).
        # Map an index in [0, n*(n-1)) to (u, v):
        #   u = idx // (n - 1), v = idx % (n - 1), and if v >= u then v += 1.
        max_c = num_src * (num_src - 1)
        if max_c <= 0:
            return 0
        max_t = torch.tensor([max_c], device=device, dtype=torch.long)
        num_t = torch.tensor([n_fam], device=device, dtype=torch.long)
        flat, _ = sampled_condensed_indices_uniformly(
            max_condensed_value=max_t,
            num_edges_to_sample=num_t,
            return_mask=False,
        )
        if flat.numel() == 0:
            return 0
        u_local = (flat // (num_src - 1)).long()
        v_local = (flat % (num_src - 1)).long()
        v_local = v_local + (v_local >= u_local).long()
        ei = torch.stack([batch_src[u_local], batch_src[v_local]], dim=0)
    else:
        max_c = num_src * num_dst
        max_t = torch.tensor([max_c], device=device, dtype=torch.long)
        num_t = torch.tensor([n_fam], device=device, dtype=torch.long)
        flat, _ = sampled_condensed_indices_uniformly(max_condensed_value=max_t, num_edges_to_sample=num_t, return_mask=False)
        si = (flat % num_dst).long().clamp(0, num_dst - 1)
        di = (flat // num_dst).long().clamp(0, num_src - 1)
        ei = torch.stack([batch_src[di], batch_dst[si]], dim=0)

    # 过滤已被其它关系族占用的边，保证 (u, v) 关系唯一。
    if hash_base is None:
        if batch_src.numel() > 0 or batch_dst.numel() > 0:
            hash_base = int(torch.max(torch.cat([batch_src, batch_dst])).item()) + 1
        else:
            hash_base = 1
    selected_src = []
    selected_dst = []
    selected_hash = set()
    for k in range(ei.shape[1]):
        s = int(ei[0, k].item())
        d = int(ei[1, k].item())
        h = s * hash_base + d
        if h in forbidden_hash or h in selected_hash:
            continue
        selected_hash.add(h)
        selected_src.append(s)
        selected_dst.append(d)

    # 若过滤后不足，补采样（拒绝采样），直到达到 n_fam 或达到尝试上限。
    max_tries = max(n_fam * 20, 1000)
    tries = 0
    same_type = src_type == dst_type
    while len(selected_src) < n_fam and tries < max_tries:
        remaining = n_fam - len(selected_src)
        cand_n = min(remaining * 10, 10000)
        src_idx = torch.randint(0, num_src, (cand_n,), device="cpu")
        if same_type:
            dst_idx = torch.randint(0, num_src, (cand_n,), device="cpu")
        else:
            dst_idx = torch.randint(0, num_dst, (cand_n,), device="cpu")
        for i in range(cand_n):
            s = int(batch_src[src_idx[i]].item())
            d = int((batch_src if same_type else batch_dst)[dst_idx[i]].item())
            if same_type and s == d:
                continue
            h = s * hash_base + d
            if h in forbidden_hash or h in selected_hash:
                continue
            selected_hash.add(h)
            selected_src.append(s)
            selected_dst.append(d)
            if len(selected_src) >= n_fam:
                break
        tries += cand_n

    if len(selected_src) == 0:
        return 0
    ei = torch.tensor([selected_src, selected_dst], dtype=torch.long, device=device)
    
    # 边类型：使用关系族内的子类别分布（如果可用），否则使用全局分布
    num_subtypes = next_offset - offset
    if edge_subtype_by_family and fam_name in edge_subtype_by_family:
        # 使用关系族内的子类别分布
        subtype_dist = edge_subtype_by_family[fam_name]
        if isinstance(subtype_dist, torch.Tensor):
            subtype_dist = subtype_dist.to(device)
        else:
            subtype_dist = torch.tensor(subtype_dist, dtype=torch.float, device=device)
        if subtype_dist.sum() > 0:
            subtype_dist = subtype_dist / subtype_dist.sum()
        else:
            subtype_dist = torch.ones(num_subtypes, dtype=torch.float, device=device) / num_subtypes
        
        # 从子类别分布中采样
        sidx = subtype_dist.multinomial(ei.shape[1], replacement=True)  # (E,)
    else:
        # 回退到关系族边际分布（优先），否则全局分布
        subtype_dist = None
        if edge_family_marginals and fam_name in edge_family_marginals:
            fam_marginals = edge_family_marginals[fam_name]
            if not isinstance(fam_marginals, torch.Tensor):
                fam_marginals = torch.tensor(fam_marginals, dtype=torch.float, device=device)
            else:
                fam_marginals = fam_marginals.to(device)
            if fam_marginals.numel() > 1:
                subtype_dist = fam_marginals[1:]
        if subtype_dist is None:
            sl = E_limit[offset:next_offset]
            if sl.numel() == 0:
                return 0
            subtype_dist = sl
        if subtype_dist.sum() > 0:
            subtype_dist = subtype_dist / subtype_dist.sum()
        else:
            subtype_dist = torch.ones(num_subtypes, dtype=torch.float, device=device) / num_subtypes
        sidx = subtype_dist.multinomial(ei.shape[1], replacement=True)
    
    global_id = (offset + sidx).long().clamp(0, out_dims_E - 1)
    all_ei.append(ei)
    all_ea.append(global_id)
    forbidden_hash.update(selected_hash)
    return int(ei.shape[1])


def compute_sparse_batched_over0_posterior_distribution(
    input_data, batch, Qt, Qsb, Qtb
):
    input_data = input_data.to(torch.float32).unsqueeze(1)  # N, 1, dt

    Qt_T = Qt[batch].transpose(-1, -2)  # N, dt, d_t-1
    left_term = input_data @ Qt_T  # N, 1, d_t-1

    right_term = Qsb[batch]  # N, d0, d_t-1
    numerator = left_term * right_term  # N, d0, d_t-1

    input_data_transposed = input_data.transpose(2, 1)  # N, dt, 1
    prod = Qtb[batch] @ input_data_transposed  # N, d0, 1
    prod = prod.squeeze(-1)  # N, d0
    denominator = prod.unsqueeze(-1)  # N, d0
    denominator[denominator == 0] = 1e-6

    out = numerator / denominator

    return out
