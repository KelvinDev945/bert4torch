import torch
import torch.nn as nn
import math
from .backend import sinusoidal_embeddings, apply_rotary_position_embeddings, sequence_masking, attention_normalize


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1, use_bias=True,
                 attention_scale=True, return_attention_scores=False, position_bias=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.attention_scale = attention_scale
        self.return_attention_scores = return_attention_scores
        self.position_bias = position_bias

        self.q = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.k = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.v = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.o = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, kv=None, position_bias=None, past_key_value=None):
        # x: [batch, seq_len, hidden]
        batch_size, seq_len = x.shape[:2]

        q = self.q(x)
        if kv is None:
            k, v = self.k(x), self.v(x)
        else:
            k, v = self.k(kv), self.v(kv)

        # split heads
        q = q.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_size).transpose(1, 2)

        # concat past_key_value for generation
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))
        if self.attention_scale:
            scores = scores / math.sqrt(self.head_size)

        # position bias
        if position_bias is not None:
            scores = scores + position_bias

        # mask
        if mask is not None:
            scores = scores + mask

        # normalize
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # context
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)

        # output
        output = self.o(context)

        outputs = [output]
        if self.return_attention_scores:
            outputs.append(attn_weights)
        if past_key_value is not None:
            outputs.append((k, v))

        return outputs if len(outputs) > 1 else outputs[0]


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


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12, conditional=False, conditional_size=None):
        super().__init__()
        self.eps = eps
        self.conditional = conditional

        if conditional:
            self.dense1 = nn.Linear(conditional_size, hidden_size, bias=False)
            self.dense2 = nn.Linear(conditional_size, hidden_size, bias=False)
        else:
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, cond=None):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        # 使用 clone() 避免 CUDA Graphs 中的 in-place 问题
        x_normalized = (x - mean) / (std + self.eps)

        if self.conditional:
            gamma = self.dense1(cond)
            beta = self.dense2(cond)
            if cond.dim() == 2:
                gamma = gamma.unsqueeze(1)
                beta = beta.unsqueeze(1)
            output = gamma * x_normalized + beta
        else:
            output = self.weight * x_normalized + self.bias
        return output


class Embedding(nn.Module):
    """可选weight共享的embedding"""
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)

    def forward(self, x):
        return self.embedding(x)


class PositionEmbedding(nn.Module):
    def __init__(self, max_position, embedding_size):
        super().__init__()
        self.embedding = nn.Embedding(max_position, embedding_size)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            return self.embedding(x[0]), self.embedding(x[1])
        seq_len = x.shape[1]
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return self.embedding(pos_ids)


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, x):
        seq_len = x.shape[1]
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0)
        return sinusoidal_embeddings(pos_ids, self.output_dim)


class RoPEPositionEmbedding(nn.Module):
    """旋转位置编码"""
    def __init__(self, embedding_size):
        super().__init__()
        self.embedding_size = embedding_size

    def forward(self, q, k):
        seq_len = q.shape[2]
        pos_ids = torch.arange(seq_len, device=q.device).unsqueeze(0)
        sinusoidal = sinusoidal_embeddings(pos_ids, self.embedding_size)
        # add heads dim
        sinusoidal = sinusoidal.unsqueeze(1)
        q, k = apply_rotary_position_embeddings(sinusoidal, q, k)
        return q, k


class RelativePositionEmbedding(nn.Module):
    """T5相对位置编码"""
    def __init__(self, num_heads, num_buckets=32, max_distance=128):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.embeddings = nn.Embedding(num_buckets, num_heads)

    def forward(self, q_len, k_len):
        q_pos = torch.arange(q_len, dtype=torch.long, device=self.embeddings.weight.device)
        k_pos = torch.arange(k_len, dtype=torch.long, device=self.embeddings.weight.device)
        rel_pos = k_pos[None, :] - q_pos[:, None]

        # convert to buckets
        num_buckets = self.num_buckets // 2
        rel_buckets = (rel_pos > 0).long() * num_buckets
        rel_pos = torch.abs(rel_pos)

        max_exact = num_buckets // 2
        is_small = rel_pos < max_exact

        rel_pos_if_large = max_exact + (
            torch.log(rel_pos.float() / max_exact) /
            math.log(self.max_distance / max_exact) *
            (num_buckets - max_exact)
        ).long()
        rel_pos_if_large = torch.min(
            rel_pos_if_large,
            torch.full_like(rel_pos_if_large, num_buckets - 1)
        )

        rel_buckets += torch.where(is_small, rel_pos, rel_pos_if_large)
        rel_embeddings = self.embeddings(rel_buckets)
        rel_embeddings = rel_embeddings.permute(2, 0, 1).unsqueeze(0)
        return rel_embeddings


class GlobalPointer(nn.Module):
    """全局指针"""
    def __init__(self, hidden_size, heads, head_size=64, use_rope=True):
        super().__init__()
        self.heads = heads
        self.head_size = head_size
        self.use_rope = use_rope
        self.dense = nn.Linear(hidden_size, heads * head_size * 2)

    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape[:2]

        # [batch, seq_len, heads*head_size*2]
        x = self.dense(x)
        x = x.view(batch_size, seq_len, self.heads, 2, self.head_size)

        # q, k: [batch, seq_len, heads, head_size]
        q, k = x[..., 0, :], x[..., 1, :]

        # RoPE
        if self.use_rope:
            q = q.transpose(1, 2)  # [batch, heads, seq_len, head_size]
            k = k.transpose(1, 2)
            pos_emb = RoPEPositionEmbedding(self.head_size)
            q, k = pos_emb(q, k)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)

        # [batch, heads, seq_len, seq_len]
        scores = torch.einsum('bmhd,bnhd->bhmn', q, k) / math.sqrt(self.head_size)

        if mask is not None:
            scores = scores + mask.unsqueeze(1)

        return scores


class CRF(nn.Module):
    """条件随机场"""
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))

    def forward(self, emissions, tags, mask=None):
        """计算loss"""
        return -self._score(emissions, tags, mask) + self._log_partition(emissions, mask)

    def decode(self, emissions, mask=None):
        """维特比解码"""
        return self._viterbi_decode(emissions, mask)

    def _score(self, emissions, tags, mask=None):
        batch_size, seq_len = tags.shape
        score = emissions[torch.arange(batch_size).unsqueeze(1), torch.arange(seq_len).unsqueeze(0), tags]

        if mask is not None:
            score = score * mask.float()

        score = score.sum(dim=1)

        # transition scores
        for i in range(seq_len - 1):
            score += self.transitions[tags[:, i], tags[:, i + 1]]

        return score.mean()

    def _log_partition(self, emissions, mask=None):
        batch_size, seq_len, num_tags = emissions.shape

        score = emissions[:, 0]
        for i in range(1, seq_len):
            score = score.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, i].unsqueeze(1)
            score = torch.logsumexp(score, dim=1)

            if mask is not None:
                score = torch.where(mask[:, i].unsqueeze(1).bool(), score, score - emissions[:, i])

        return torch.logsumexp(score, dim=1).mean()

    def _viterbi_decode(self, emissions, mask=None):
        batch_size, seq_len, num_tags = emissions.shape

        score = emissions[:, 0]
        history = []

        for i in range(1, seq_len):
            score_with_trans = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            max_scores, max_ids = score_with_trans.max(dim=1)
            score = max_scores + emissions[:, i]
            history.append(max_ids)

        # backtrack
        best_tags_list = []
        _, best_last_tag = score.max(dim=1)
        best_tags = [best_last_tag]

        for hist in reversed(history):
            best_last_tag = hist[torch.arange(batch_size), best_last_tag]
            best_tags.append(best_last_tag)

        best_tags.reverse()
        return torch.stack(best_tags, dim=1)
