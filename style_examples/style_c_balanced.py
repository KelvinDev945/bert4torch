"""
风格 C：混合风格（推荐）
- 保持代码简洁，但遵循 PyTorch 最佳实践
- 关键位置的类型注解（主要在公共接口）
- 适度的文档字符串
- 平衡可读性、简洁性和规范性
"""
import torch
import torch.nn as nn
from typing import Optional
import math


class MultiHeadAttention(nn.Module):
    """多头注意力机制

    Args:
        hidden_size: 隐藏层维度
        num_attention_heads: 注意力头数
        attention_dropout: 注意力 dropout 概率，默认 0.1
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.1
    ):
        super().__init__()
        assert hidden_size % num_attention_heads == 0, \
            f"hidden_size ({hidden_size}) 必须能被 num_attention_heads ({num_attention_heads}) 整除"

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        # Q, K, V 投影
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.o = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(attention_dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """将最后一维拆分为 (num_heads, head_size)"""
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        return x.transpose(1, 2)  # [batch, heads, seq_len, head_size]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """合并多个注意力头"""
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ):
        """
        Args:
            x: 输入张量 [batch_size, seq_len, hidden_size]
            mask: 注意力掩码 [batch_size, 1, 1, seq_len] 或 [batch_size, 1, seq_len, seq_len]
            return_attention_weights: 是否返回注意力权重

        Returns:
            输出张量 [batch_size, seq_len, hidden_size]
            注意力权重（可选）
        """
        # Linear projections and split heads
        q = self._split_heads(self.q(x))  # [batch, heads, seq_len, head_size]
        k = self._split_heads(self.k(x))
        v = self._split_heads(self.v(x))

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.attention_head_size)

        if mask is not None:
            scores = scores + mask  # mask 中 0 的位置会被加上负无穷

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values and merge heads
        context = torch.matmul(attn_weights, v)
        context = self._merge_heads(context)

        # Output projection
        output = self.o(context)

        if return_attention_weights:
            return output, attn_weights
        return output


class FeedForward(nn.Module):
    """位置前馈网络（两层全连接 + 激活函数）

    Args:
        hidden_size: 隐藏层维度
        intermediate_size: 中间层维度
        dropout: dropout 概率，默认 0.1
        activation: 激活函数类型，默认 'gelu'
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        # 激活函数
        activation_map = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'swish': nn.SiLU(),
        }
        self.activation = activation_map.get(activation, nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [batch_size, seq_len, hidden_size]

        Returns:
            输出张量 [batch_size, seq_len, hidden_size]
        """
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x


class TransformerLayer(nn.Module):
    """标准 Transformer 层（Self-Attention + FeedForward）

    Args:
        hidden_size: 隐藏层维度
        num_attention_heads: 注意力头数
        intermediate_size: FFN 中间层维度
        dropout: dropout 概率
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        intermediate_size: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_size, num_attention_heads, dropout)
        self.feedforward = FeedForward(hidden_size, intermediate_size, dropout)

        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: 输入张量 [batch_size, seq_len, hidden_size]
            mask: 注意力掩码

        Returns:
            输出张量 [batch_size, seq_len, hidden_size]
        """
        # Self-Attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # FeedForward with residual connection
        ff_output = self.feedforward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


# 测试代码
if __name__ == '__main__':
    print("=" * 60)
    print("风格 C：混合风格示例测试")
    print("=" * 60)

    batch_size, seq_len, hidden_size = 2, 10, 768
    x = torch.randn(batch_size, seq_len, hidden_size)

    # 测试 MultiHeadAttention
    print("\n[1] 测试 MultiHeadAttention")
    attn = MultiHeadAttention(hidden_size=768, num_attention_heads=12)
    output = attn(x)
    print(f"  输入: {x.shape} -> 输出: {output.shape}")

    # 测试 FeedForward
    print("\n[2] 测试 FeedForward")
    ff = FeedForward(hidden_size=768, intermediate_size=3072)
    output = ff(x)
    print(f"  输入: {x.shape} -> 输出: {output.shape}")

    # 测试完整的 TransformerLayer
    print("\n[3] 测试 TransformerLayer（Attention + FFN）")
    layer = TransformerLayer(
        hidden_size=768,
        num_attention_heads=12,
        intermediate_size=3072
    )
    output = layer(x)
    print(f"  输入: {x.shape} -> 输出: {output.shape}")

    # 测试带 mask 的情况
    print("\n[4] 测试带注意力掩码")
    # 创建一个简单的 padding mask
    mask = torch.ones(batch_size, 1, 1, seq_len)
    mask[:, :, :, 5:] = 0  # 屏蔽后 5 个位置
    mask = (1.0 - mask) * -10000.0  # 转换为 additive mask
    output = layer(x, mask)
    print(f"  输入: {x.shape}, Mask: {mask.shape} -> 输出: {output.shape}")

    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)

    print("\n【风格 C 的特点】")
    print("• 代码简洁但不失规范")
    print("• 主要接口有类型注解")
    print("• 文档字符串简洁明了")
    print("• 包含完整的 TransformerLayer 示例")
    print("• 适合快速开发和迭代")
