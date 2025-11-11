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
