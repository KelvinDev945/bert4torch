"""backend.py 单元测试"""
import unittest
import torch
import sys
sys.path.insert(0, '..')

from bert4torch.backend import (
    gelu,
    sinusoidal_embeddings,
    apply_rotary_position_embeddings,
    sequence_masking,
    attention_normalize,
    piecewise_linear
)


class TestBackend(unittest.TestCase):
    """测试 backend.py 中的工具函数"""

    def test_gelu(self):
        """测试 GELU 激活函数"""
        x = torch.randn(2, 3, 4)
        output = gelu(x)

        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.is_tensor(output))

        # 测试特殊值
        self.assertAlmostEqual(gelu(torch.tensor(0.0)).item(), 0.0, places=5)

    def test_sinusoidal_embeddings(self):
        """测试正弦位置编码"""
        batch_size, seq_len, dim = 2, 10, 64
        pos_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)

        embeddings = sinusoidal_embeddings(pos_ids, dim)

        self.assertEqual(embeddings.shape, (batch_size, seq_len, dim))

        # 位置0应该是特定模式
        self.assertAlmostEqual(embeddings[0, 0, 0].item(), 0.0, places=5)

    def test_apply_rotary_position_embeddings(self):
        """测试旋转位置编码"""
        batch_size, num_heads, seq_len, head_size = 2, 12, 10, 64

        q = torch.randn(batch_size, num_heads, seq_len, head_size)
        k = torch.randn(batch_size, num_heads, seq_len, head_size)

        pos_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, seq_len)
        sinusoidal = sinusoidal_embeddings(pos_ids, head_size).unsqueeze(1)

        q_rot, k_rot = apply_rotary_position_embeddings(sinusoidal, q, k)

        self.assertEqual(q_rot.shape, q.shape)
        self.assertEqual(k_rot.shape, k.shape)

    def test_sequence_masking(self):
        """测试序列mask"""
        batch_size, seq_len, hidden_size = 2, 10, 768
        x = torch.randn(batch_size, seq_len, hidden_size)
        mask = torch.ones(batch_size, seq_len)
        mask[:, 5:] = 0  # 后半部分mask掉

        output = sequence_masking(x, mask, value=0.0)

        self.assertEqual(output.shape, x.shape)
        # 检查mask后的部分是否为0
        self.assertTrue(torch.allclose(output[:, 5:, :], torch.zeros_like(output[:, 5:, :])))
        # 检查未mask的部分是否保持不变
        self.assertTrue(torch.allclose(output[:, :5, :], x[:, :5, :]))

    def test_attention_normalize_softmax(self):
        """测试 attention 归一化（softmax）"""
        batch_size, num_heads, seq_len = 2, 12, 10
        scores = torch.randn(batch_size, num_heads, seq_len, seq_len)

        normalized = attention_normalize(scores, method='softmax')

        self.assertEqual(normalized.shape, scores.shape)
        # 检查每行和为1
        row_sums = normalized.sum(dim=-1)
        self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums)))

    def test_attention_normalize_squared_relu(self):
        """测试 attention 归一化（squared relu）"""
        batch_size, num_heads, seq_len = 2, 12, 10
        scores = torch.randn(batch_size, num_heads, seq_len, seq_len)

        normalized = attention_normalize(scores, method='squared_relu')

        self.assertEqual(normalized.shape, scores.shape)
        # 所有值应该非负
        self.assertTrue((normalized >= 0).all())

    def test_piecewise_linear(self):
        """测试分段线性函数"""
        schedule = {0: 0.0, 1000: 1.0, 2000: 0.5, 3000: 0.0}

        # 测试边界值
        self.assertEqual(piecewise_linear(0, schedule), 0.0)
        self.assertEqual(piecewise_linear(1000, schedule), 1.0)
        self.assertEqual(piecewise_linear(3000, schedule), 0.0)

        # 测试中间值
        self.assertAlmostEqual(piecewise_linear(500, schedule), 0.5, places=5)
        self.assertAlmostEqual(piecewise_linear(1500, schedule), 0.75, places=5)

        # 测试超出范围
        self.assertEqual(piecewise_linear(-100, schedule), 0.0)
        self.assertEqual(piecewise_linear(5000, schedule), 0.0)


if __name__ == '__main__':
    unittest.main()
