"""layers.py 单元测试"""
import unittest
import torch
import sys
sys.path.insert(0, '..')

from bert4torch.layers import (
    MultiHeadAttention,
    FeedForward,
    LayerNorm,
    Embedding,
    PositionEmbedding,
    SinusoidalPositionEmbedding,
    RoPEPositionEmbedding,
    GlobalPointer,
    CRF
)


class TestLayers(unittest.TestCase):
    """测试 layers.py 中的层"""

    def test_multihead_attention(self):
        """测试多头注意力"""
        batch_size, seq_len, hidden_size = 2, 10, 768
        num_heads = 12

        attn = MultiHeadAttention(hidden_size, num_heads)
        x = torch.randn(batch_size, seq_len, hidden_size)

        output = attn(x)

        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))

    def test_multihead_attention_with_mask(self):
        """测试带mask的多头注意力"""
        batch_size, seq_len, hidden_size = 2, 10, 768
        num_heads = 12

        attn = MultiHeadAttention(hidden_size, num_heads)
        x = torch.randn(batch_size, seq_len, hidden_size)
        mask = torch.zeros(batch_size, 1, 1, seq_len)
        mask[:, :, :, 5:] = -10000.0  # mask后半部分

        output = attn(x, mask=mask)

        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))

    def test_multihead_attention_cross_attention(self):
        """测试交叉注意力"""
        batch_size, q_len, k_len, hidden_size = 2, 10, 15, 768
        num_heads = 12

        attn = MultiHeadAttention(hidden_size, num_heads)
        q = torch.randn(batch_size, q_len, hidden_size)
        kv = torch.randn(batch_size, k_len, hidden_size)

        output = attn(q, kv=kv)

        self.assertEqual(output.shape, (batch_size, q_len, hidden_size))

    def test_feedforward(self):
        """测试前馈网络"""
        batch_size, seq_len, hidden_size = 2, 10, 768
        intermediate_size = 3072

        ffn = FeedForward(hidden_size, intermediate_size)
        x = torch.randn(batch_size, seq_len, hidden_size)

        output = ffn(x)

        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))

    def test_layernorm(self):
        """测试层归一化"""
        batch_size, seq_len, hidden_size = 2, 10, 768

        ln = LayerNorm(hidden_size)
        x = torch.randn(batch_size, seq_len, hidden_size)

        output = ln(x)

        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))

        # 检查归一化后的统计特性
        mean = output.mean(dim=-1)
        std = output.std(dim=-1)

        self.assertTrue(torch.allclose(mean, torch.zeros_like(mean), atol=1e-5))
        self.assertTrue(torch.allclose(std, torch.ones_like(std), atol=1e-5))

    def test_layernorm_conditional(self):
        """测试条件层归一化"""
        batch_size, seq_len, hidden_size = 2, 10, 768
        cond_size = 128

        ln = LayerNorm(hidden_size, conditional=True, conditional_size=cond_size)
        x = torch.randn(batch_size, seq_len, hidden_size)
        cond = torch.randn(batch_size, cond_size)

        output = ln(x, cond=cond)

        self.assertEqual(output.shape, (batch_size, seq_len, hidden_size))

    def test_embedding(self):
        """测试嵌入层"""
        batch_size, seq_len = 2, 10
        vocab_size, embedding_size = 1000, 768

        emb = Embedding(vocab_size, embedding_size)
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        output = emb(token_ids)

        self.assertEqual(output.shape, (batch_size, seq_len, embedding_size))

    def test_position_embedding(self):
        """测试位置嵌入"""
        batch_size, seq_len = 2, 10
        max_position, embedding_size = 512, 768

        pos_emb = PositionEmbedding(max_position, embedding_size)
        token_ids = torch.randint(0, 1000, (batch_size, seq_len))

        output = pos_emb(token_ids)

        self.assertEqual(output.shape, (batch_size, seq_len, embedding_size))

    def test_sinusoidal_position_embedding(self):
        """测试正弦位置嵌入"""
        batch_size, seq_len = 2, 10
        output_dim = 768

        pos_emb = SinusoidalPositionEmbedding(output_dim)
        x = torch.randn(batch_size, seq_len, output_dim)

        output = pos_emb(x)

        self.assertEqual(output.shape, (batch_size, seq_len, output_dim))

    def test_rope_position_embedding(self):
        """测试RoPE位置嵌入"""
        batch_size, num_heads, seq_len, head_size = 2, 12, 10, 64

        rope = RoPEPositionEmbedding(head_size)
        q = torch.randn(batch_size, num_heads, seq_len, head_size)
        k = torch.randn(batch_size, num_heads, seq_len, head_size)

        q_rot, k_rot = rope(q, k)

        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)

    def test_global_pointer(self):
        """测试GlobalPointer"""
        batch_size, seq_len, hidden_size = 2, 10, 768
        heads = 4
        head_size = 64

        gp = GlobalPointer(hidden_size, heads, head_size)
        x = torch.randn(batch_size, seq_len, hidden_size)

        scores = gp(x)

        self.assertEqual(scores.shape, (batch_size, heads, seq_len, seq_len))

    def test_crf_forward(self):
        """测试CRF前向传播"""
        batch_size, seq_len, num_tags = 2, 10, 7

        crf = CRF(num_tags)
        emissions = torch.randn(batch_size, seq_len, num_tags)
        tags = torch.randint(0, num_tags, (batch_size, seq_len))

        loss = crf(emissions, tags)

        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.shape, ())  # scalar

    def test_crf_decode(self):
        """测试CRF解码"""
        batch_size, seq_len, num_tags = 2, 10, 7

        crf = CRF(num_tags)
        emissions = torch.randn(batch_size, seq_len, num_tags)

        predictions = crf.decode(emissions)

        self.assertEqual(predictions.shape, (batch_size, seq_len))
        # 检查预测值在合法范围内
        self.assertTrue((predictions >= 0).all())
        self.assertTrue((predictions < num_tags).all())


class TestLayerShapes(unittest.TestCase):
    """测试层的形状变换"""

    def test_attention_head_split(self):
        """测试注意力头的拆分和合并"""
        batch_size, seq_len, hidden_size = 2, 10, 768
        num_heads = 12
        head_size = hidden_size // num_heads

        attn = MultiHeadAttention(hidden_size, num_heads)

        # 测试内部形状变换是否正确
        x = torch.randn(batch_size, seq_len, hidden_size)
        output = attn(x)

        # 输出形状应该和输入一致
        self.assertEqual(output.shape, x.shape)


if __name__ == '__main__':
    unittest.main()
