import torch
import torch.nn as nn
import math


def gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def sinusoidal_embeddings(pos_ids, output_dim):
    """正弦位置编码"""
    indices = torch.arange(0, output_dim // 2, dtype=torch.float, device=pos_ids.device)
    indices = torch.pow(10000.0, -2 * indices / output_dim)
    embeddings = torch.einsum('bn,d->bnd', pos_ids.float(), indices)
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
    embeddings = embeddings.reshape(*pos_ids.shape, output_dim)
    return embeddings


def apply_rotary_position_embeddings(sinusoidal, *tensors):
    """旋转位置编码 RoPE"""
    assert len(tensors) > 0

    sin, cos = sinusoidal[..., 0::2], sinusoidal[..., 1::2]
    sin, cos = torch.cat([sin, sin], dim=-1), torch.cat([cos, cos], dim=-1)

    def rotate(x):
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1)
        x2 = x2.reshape(*x.shape)
        return x * cos + x2 * sin

    return [rotate(t) for t in tensors]


def yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    """YaRN: 找到需要修正的维度

    Args:
        num_rotations: 旋转次数（beta 的个数）
        dim: 隐藏层维度
        base: RoPE 基数
        max_position_embeddings: 最大位置数

    Returns:
        修正维度索引
    """
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    """YaRN: 找到修正范围

    Args:
        low_rot: 低旋转次数
        high_rot: 高旋转次数
        dim: 隐藏层维度
        base: RoPE 基数
        max_position_embeddings: 最大位置数

    Returns:
        (low, high) 修正范围
    """
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings))
    high = math.ceil(yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim - 1)


def yarn_linear_ramp_mask(min_val, max_val, dim):
    """YaRN: 线性斜坡掩码

    Args:
        min_val: 最小值
        max_val: 最大值
        dim: 维度

    Returns:
        斜坡掩码张量
    """
    if min_val == max_val:
        max_val += 0.001

    linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
    return torch.clamp(linear_func, 0, 1)


def yarn_get_mscale(scale=1.0, mscale=1.0):
    """YaRN: 获取注意力缩放因子

    Args:
        scale: RoPE 缩放因子
        mscale: 缩放系数

    Returns:
        注意力缩放因子
    """
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_sinusoidal_embeddings(
    pos_ids,
    output_dim,
    base=10000,
    scale=1.0,
    original_max_position_embeddings=2048,
    extrapolation_factor=1.0,
    attn_factor=1.0,
    beta_fast=32,
    beta_slow=1,
    mscale=1.0,
    mscale_all_dim=0.0
):
    """YaRN 位置编码

    参考: https://arxiv.org/abs/2309.00071
    YaRN (Yet another RoPE extensioN) 通过动态缩放和 NTK 插值来扩展上下文长度

    Args:
        pos_ids: 位置 ID
        output_dim: 输出维度
        base: RoPE 基数
        scale: 缩放因子（扩展倍数）
        original_max_position_embeddings: 原始最大位置数
        extrapolation_factor: 外推因子
        attn_factor: 注意力因子
        beta_fast: 快速旋转阈值
        beta_slow: 慢速旋转阈值
        mscale: 注意力缩放系数
        mscale_all_dim: 所有维度的缩放

    Returns:
        YaRN 位置编码张量
    """
    device = pos_ids.device
    dim = output_dim // 2

    # 计算频率
    freq_extra = 1.0 / (base ** (torch.arange(0, dim, dtype=torch.float32, device=device) / dim))

    # NTK-Aware 缩放
    freq_inter = 1.0 / (
        (scale * base) ** (torch.arange(0, dim, dtype=torch.float32, device=device) / dim)
    )

    # 找到需要插值的范围
    low, high = yarn_find_correction_range(
        beta_fast, beta_slow, dim, base, original_max_position_embeddings
    )

    # 计算插值掩码
    inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim).to(device)
    inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

    # 应用缩放
    inv_freq = inv_freq / extrapolation_factor

    # 计算位置编码
    embeddings = torch.einsum('bn,d->bnd', pos_ids.float(), inv_freq)

    # 计算 mscale
    if mscale != 1.0:
        mscale_val = yarn_get_mscale(scale, mscale) / yarn_get_mscale(scale, mscale_all_dim)
        embeddings = embeddings * mscale_val * attn_factor

    # 生成 sin 和 cos
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
    embeddings = embeddings.reshape(*pos_ids.shape, output_dim)

    return embeddings


def apply_yarn_rotary_embeddings(
    sinusoidal,
    *tensors,
    scale=1.0,
    mscale=1.0
):
    """应用 YaRN 旋转位置编码

    Args:
        sinusoidal: YaRN 正弦位置编码
        *tensors: 要应用编码的张量（如 q, k）
        scale: 缩放因子
        mscale: 注意力缩放系数

    Returns:
        应用了 YaRN 编码的张量列表
    """
    assert len(tensors) > 0

    sin, cos = sinusoidal[..., 0::2], sinusoidal[..., 1::2]
    sin, cos = torch.cat([sin, sin], dim=-1), torch.cat([cos, cos], dim=-1)

    def rotate(x):
        x2 = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1)
        x2 = x2.reshape(*x.shape)
        return x * cos + x2 * sin

    return [rotate(t) for t in tensors]


def sequence_masking(x, mask, value=0.0, axis=None):
    """序列mask"""
    if mask is None:
        return x
    if axis is None:
        axis = 1
    if axis < 0:
        axis = x.dim() + axis

    assert mask.dim() == 2
    for _ in range(axis - 1):
        mask = mask.unsqueeze(1)
    for _ in range(x.dim() - mask.dim()):
        mask = mask.unsqueeze(-1)

    return x * mask.float() + value * (1 - mask.float())


def attention_normalize(a, mask=None, axis=-1, method='softmax'):
    """attention归一化"""
    if method == 'softmax':
        return torch.softmax(a, dim=axis)
    elif method == 'squared_relu':
        mask_value = -1e12 if mask is None else 0.0
        a = sequence_masking(a, mask, mask_value, axis - 1)
        return torch.relu(a) ** 2
    else:
        raise ValueError(f'Unknown method: {method}')


def piecewise_linear(step, schedule):
    """分段线性函数"""
    schedule = sorted(schedule.items())
    if step <= schedule[0][0]:
        return schedule[0][1]
    elif step >= schedule[-1][0]:
        return schedule[-1][1]
    else:
        for i in range(len(schedule) - 1):
            t1, v1 = schedule[i]
            t2, v2 = schedule[i + 1]
            if t1 <= step <= t2:
                k = (step - t1) / (t2 - t1)
                return v1 + k * (v2 - v1)
