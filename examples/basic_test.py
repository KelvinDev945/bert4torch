"""基础功能测试"""
import torch
import sys
sys.path.insert(0, '..')

from bert4torch.models import BERT, RoFormer, GPT, T5
from bert4torch.layers import MultiHeadAttention, FeedForward


def test_layers():
    """测试基础层"""
    print("=" * 60)
    print("测试基础层")
    print("=" * 60)

    batch_size, seq_len, hidden_size = 2, 10, 768
    x = torch.randn(batch_size, seq_len, hidden_size)

    # Test MultiHeadAttention
    print("\n[1] MultiHeadAttention")
    attn = MultiHeadAttention(hidden_size=768, num_heads=12)
    output = attn(x)
    print(f"  Input: {x.shape} -> Output: {output.shape}")
    assert output.shape == x.shape

    # Test FeedForward
    print("\n[2] FeedForward")
    ffn = FeedForward(hidden_size=768, intermediate_size=3072)
    output = ffn(x)
    print(f"  Input: {x.shape} -> Output: {output.shape}")
    assert output.shape == x.shape

    print("\n✓ 基础层测试通过!")


def test_bert():
    """测试BERT模型"""
    print("\n" + "=" * 60)
    print("测试BERT模型")
    print("=" * 60)

    batch_size, seq_len = 2, 128
    vocab_size, hidden_size = 21128, 768

    model = BERT(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072
    )

    # 测试前向传播
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    segment_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)

    output = model(token_ids, segment_ids)
    print(f"\nBERT输出: {output.shape}")
    print(f"期望: [{batch_size}, {seq_len}, {hidden_size}]")
    assert output.shape == (batch_size, seq_len, hidden_size)

    print("\n✓ BERT模型测试通过!")


def test_roformer():
    """测试RoFormer模型"""
    print("\n" + "=" * 60)
    print("测试RoFormer模型")
    print("=" * 60)

    batch_size, seq_len = 2, 128
    vocab_size, hidden_size = 21128, 768

    model = RoFormer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072
    )

    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = model(token_ids)

    print(f"\nRoFormer输出: {output.shape}")
    assert output.shape == (batch_size, seq_len, hidden_size)

    print("\n✓ RoFormer模型测试通过!")


def test_gpt():
    """测试GPT模型"""
    print("\n" + "=" * 60)
    print("测试GPT模型")
    print("=" * 60)

    batch_size, seq_len = 2, 128
    vocab_size, hidden_size = 50257, 768

    model = GPT(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072
    )

    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = model(token_ids)

    print(f"\nGPT输出: {output.shape}")
    assert output.shape == (batch_size, seq_len, hidden_size)

    print("\n✓ GPT模型测试通过!")


def test_t5():
    """测试T5模型"""
    print("\n" + "=" * 60)
    print("测试T5模型")
    print("=" * 60)

    batch_size, seq_len = 2, 128
    vocab_size, hidden_size = 32128, 768

    model = T5(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072
    )

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    decoder_input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    output = model(input_ids, decoder_input_ids)

    print(f"\nT5输出: {output.shape}")
    print(f"期望: [{batch_size}, {seq_len}, {vocab_size}]")
    assert output.shape == (batch_size, seq_len, vocab_size)

    print("\n✓ T5模型测试通过!")


if __name__ == '__main__':
    test_layers()
    test_bert()
    test_roformer()
    test_gpt()
    test_t5()

    print("\n" + "=" * 60)
    print("🎉 所有测试通过!")
    print("=" * 60)
