"""
风格 A：bert4keras 简洁风格
- 最小依赖，单文件可运行
- 代码简洁，易于阅读和修改
- 保持与 bert4keras 类似的 API 设计
"""
import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.o = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, x, mask=None):
        # x: [batch, seq_len, hidden]
        batch_size, seq_len = x.shape[:2]

        # Linear projections
        q = self.q(x)  # [batch, seq_len, hidden]
        k = self.k(x)
        v = self.v(x)

        # Split heads: [batch, seq_len, hidden] -> [batch, heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.attention_head_size)

        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax and dropout
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, v)

        # Merge heads: [batch, heads, seq_len, head_dim] -> [batch, seq_len, hidden]
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # Final linear
        output = self.o(context)
        return output


class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.1, activation='gelu'):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == 'gelu' else nn.ReLU()

    def forward(self, x):
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x


# 使用示例
if __name__ == '__main__':
    batch_size = 2
    seq_len = 10
    hidden_size = 768

    # 创建模型
    attn = MultiHeadAttention(hidden_size=768, num_attention_heads=12)
    ff = FeedForward(hidden_size=768, intermediate_size=3072)

    # 测试数据
    x = torch.randn(batch_size, seq_len, hidden_size)

    # 前向传播
    attn_out = attn(x)
    ff_out = ff(attn_out)

    print(f"Input shape: {x.shape}")
    print(f"Attention output shape: {attn_out.shape}")
    print(f"FeedForward output shape: {ff_out.shape}")
