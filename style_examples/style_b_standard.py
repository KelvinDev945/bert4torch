"""
风格 B：标准 PyTorch 风格
- 完整的类型注解和文档字符串
- 详细的参数说明和返回值说明
- 遵循 PyTorch 官方代码规范
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism.

    This implementation follows the standard multi-head attention as described in
    "Attention is All You Need" (Vaswani et al., 2017).

    Args:
        hidden_size (int): The dimensionality of input and output features.
        num_attention_heads (int): The number of attention heads.
        attention_dropout (float, optional): Dropout probability for attention weights.
            Default: 0.1
        bias (bool, optional): Whether to use bias in linear projections. Default: True

    Attributes:
        hidden_size (int): The dimensionality of input and output features.
        num_attention_heads (int): The number of attention heads.
        attention_head_size (int): The dimensionality of each attention head.
        all_head_size (int): Total size of all attention heads.

    Example:
        >>> attn = MultiHeadAttention(hidden_size=768, num_attention_heads=12)
        >>> x = torch.randn(2, 10, 768)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([2, 10, 768])
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.1,
        bias: bool = True
    ) -> None:
        super().__init__()

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by "
                f"num_attention_heads ({num_attention_heads})"
            )

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # Linear projections for Q, K, V
        self.query = nn.Linear(hidden_size, self.all_head_size, bias=bias)
        self.key = nn.Linear(hidden_size, self.all_head_size, bias=bias)
        self.value = nn.Linear(hidden_size, self.all_head_size, bias=bias)

        # Output projection
        self.output_projection = nn.Linear(self.all_head_size, hidden_size, bias=bias)

        # Dropout
        self.attention_dropout = nn.Dropout(attention_dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Split the last dimension into (num_attention_heads, attention_head_size).

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, num_heads, seq_len, head_size]
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge attention heads.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_heads, seq_len, head_size]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.all_head_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of multi-head attention.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]
            attention_mask (torch.Tensor, optional): Attention mask of shape
                [batch_size, 1, 1, seq_len] or [batch_size, 1, seq_len, seq_len].
                Positions with value 0 will be masked. Default: None
            return_attention_weights (bool, optional): Whether to return attention weights.
                Default: False

        Returns:
            tuple: A tuple containing:
                - output (torch.Tensor): Output tensor of shape [batch_size, seq_len, hidden_size]
                - attention_weights (torch.Tensor, optional): Attention weights if
                  return_attention_weights is True
        """
        batch_size, seq_len, _ = hidden_states.size()

        # Linear projections
        query_layer = self.query(hidden_states)  # [batch, seq_len, all_head_size]
        key_layer = self.key(hidden_states)
        value_layer = self.value(hidden_states)

        # Split into multiple heads
        query_layer = self._split_heads(query_layer)  # [batch, heads, seq_len, head_size]
        key_layer = self._split_heads(key_layer)
        value_layer = self._split_heads(value_layer)

        # Compute attention scores
        # [batch, heads, seq_len, seq_len]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply attention mask
        if attention_mask is not None:
            # attention_mask: [batch, 1, 1, seq_len] or [batch, 1, seq_len, seq_len]
            attention_scores = attention_scores + attention_mask

        # Normalize attention scores to probabilities
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # Apply dropout
        attention_probs = self.attention_dropout(attention_probs)

        # Compute context layer
        context_layer = torch.matmul(attention_probs, value_layer)  # [batch, heads, seq_len, head_size]

        # Merge heads
        context_layer = self._merge_heads(context_layer)  # [batch, seq_len, all_head_size]

        # Final output projection
        output = self.output_projection(context_layer)  # [batch, seq_len, hidden_size]

        if return_attention_weights:
            return output, attention_probs
        else:
            return output, None


class FeedForward(nn.Module):
    """Position-wise feed-forward network.

    Applies two linear transformations with an activation function in between.

    Args:
        hidden_size (int): The dimensionality of input and output features.
        intermediate_size (int): The dimensionality of the intermediate layer.
        hidden_dropout (float, optional): Dropout probability. Default: 0.1
        activation (str, optional): Activation function name. Options: 'gelu', 'relu'.
            Default: 'gelu'

    Example:
        >>> ff = FeedForward(hidden_size=768, intermediate_size=3072)
        >>> x = torch.randn(2, 10, 768)
        >>> output = ff(x)
        >>> print(output.shape)
        torch.Size([2, 10, 768])
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_dropout: float = 0.1,
        activation: str = 'gelu'
    ) -> None:
        super().__init__()

        self.dense_1 = nn.Linear(hidden_size, intermediate_size)
        self.dense_2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout)

        # Select activation function
        if activation == 'gelu':
            self.activation_fn = nn.GELU()
        elif activation == 'relu':
            self.activation_fn = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Forward pass of feed-forward network.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_size]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, hidden_size]
        """
        hidden_states = self.dense_1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense_2(hidden_states)
        return hidden_states


# Unit test
if __name__ == '__main__':
    # Test MultiHeadAttention
    print("Testing MultiHeadAttention...")
    attn = MultiHeadAttention(hidden_size=768, num_attention_heads=12)
    x = torch.randn(2, 10, 768)
    output, _ = attn(x)
    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
    print(f"✓ MultiHeadAttention test passed. Output shape: {output.shape}")

    # Test FeedForward
    print("\nTesting FeedForward...")
    ff = FeedForward(hidden_size=768, intermediate_size=3072)
    output = ff(x)
    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
    print(f"✓ FeedForward test passed. Output shape: {output.shape}")

    print("\n✓ All tests passed!")
