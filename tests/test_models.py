"""models.py 单元测试"""
import unittest
import torch
import sys
sys.path.insert(0, '..')

from bert4torch.models import BERT, RoFormer, GPT, T5, build_transformer_model


class TestModels(unittest.TestCase):
    """测试 models.py 中的模型"""

    def test_bert_basic(self):
        """测试BERT基础功能"""
        vocab_size = 1000
        batch_size, seq_len = 2, 10

        model = BERT(
            vocab_size=vocab_size,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024
        )

        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        segment_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

        output = model(token_ids, segment_ids)

        self.assertEqual(output.shape, (batch_size, seq_len, 256))

    def test_bert_with_pool(self):
        """测试BERT带pooler"""
        vocab_size = 1000
        batch_size, seq_len = 2, 10

        model = BERT(
            vocab_size=vocab_size,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024,
            with_pool=True
        )

        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        segment_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

        hidden, pooled = model(token_ids, segment_ids)

        self.assertEqual(hidden.shape, (batch_size, seq_len, 256))
        self.assertEqual(pooled.shape, (batch_size, 256))

    def test_bert_with_mlm(self):
        """测试BERT MLM"""
        vocab_size = 1000
        batch_size, seq_len = 2, 10

        model = BERT(
            vocab_size=vocab_size,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024,
            with_mlm=True
        )

        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        segment_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

        mlm_scores = model(token_ids, segment_ids)

        self.assertEqual(mlm_scores.shape, (batch_size, seq_len, vocab_size))

    def test_roformer(self):
        """测试RoFormer"""
        vocab_size = 1000
        batch_size, seq_len = 2, 10

        model = RoFormer(
            vocab_size=vocab_size,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024
        )

        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        output = model(token_ids)

        self.assertEqual(output.shape, (batch_size, seq_len, 256))

    def test_gpt(self):
        """测试GPT"""
        vocab_size = 1000
        batch_size, seq_len = 2, 10

        model = GPT(
            vocab_size=vocab_size,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024
        )

        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        output = model(token_ids)

        self.assertEqual(output.shape, (batch_size, seq_len, 256))

    def test_t5(self):
        """测试T5"""
        vocab_size = 1000
        batch_size, seq_len = 2, 10

        model = T5(
            vocab_size=vocab_size,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024
        )

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        decoder_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        output = model(input_ids, decoder_input_ids)

        self.assertEqual(output.shape, (batch_size, seq_len, vocab_size))

    def test_t5_encode_decode(self):
        """测试T5编码解码"""
        vocab_size = 1000
        batch_size, seq_len = 2, 10

        model = T5(
            vocab_size=vocab_size,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024
        )

        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        # 测试编码
        encoder_output = model.encode(input_ids)
        self.assertEqual(encoder_output.shape, (batch_size, seq_len, 256))

        # 测试解码
        decoder_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        decoder_output = model.decode(decoder_input_ids, encoder_output)
        self.assertEqual(decoder_output.shape, (batch_size, seq_len, 256))


class TestModelConsistency(unittest.TestCase):
    """测试模型一致性"""

    def test_bert_deterministic(self):
        """测试BERT的确定性（相同输入应得到相同输出）"""
        vocab_size = 1000
        batch_size, seq_len = 2, 10

        model = BERT(
            vocab_size=vocab_size,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512
        )
        model.eval()

        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        segment_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

        with torch.no_grad():
            output1 = model(token_ids, segment_ids)
            output2 = model(token_ids, segment_ids)

        self.assertTrue(torch.allclose(output1, output2))

    def test_model_gradient_flow(self):
        """测试模型梯度流"""
        vocab_size = 1000
        batch_size, seq_len = 2, 10

        model = BERT(
            vocab_size=vocab_size,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512,
            with_mlm=True
        )

        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        segment_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # 前向传播
        output = model(token_ids, segment_ids)

        # 计算loss
        loss = torch.nn.functional.cross_entropy(
            output.view(-1, vocab_size),
            labels.view(-1)
        )

        # 反向传播
        loss.backward()

        # 检查梯度是否存在
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                break

        self.assertTrue(has_grad, "模型应该有梯度")


class TestModelMasks(unittest.TestCase):
    """测试模型的mask功能"""

    def test_bert_attention_mask(self):
        """测试BERT的attention mask"""
        vocab_size = 1000
        batch_size, seq_len = 2, 10

        model = BERT(
            vocab_size=vocab_size,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512
        )
        model.eval()

        # 创建带padding的输入
        token_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
        token_ids[:, 5:] = 0  # 后半部分padding

        segment_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

        with torch.no_grad():
            output = model(token_ids, segment_ids)

        # 输出应该有正确的形状
        self.assertEqual(output.shape, (batch_size, seq_len, 256))

    def test_gpt_causal_mask(self):
        """测试GPT的因果mask"""
        vocab_size = 1000
        batch_size, seq_len = 2, 10

        model = GPT(
            vocab_size=vocab_size,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=512
        )
        model.eval()

        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            output = model(token_ids)

        # GPT应该能够正常处理
        self.assertEqual(output.shape, (batch_size, seq_len, 256))


if __name__ == '__main__':
    unittest.main()
