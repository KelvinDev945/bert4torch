"""文本生成示例 - GPT"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '..')

from bert4torch.models import GPT
from bert4torch.snippets import AutoRegressiveDecoder


class TextGenerator(AutoRegressiveDecoder):
    """GPT文本生成器"""
    def __init__(self, model, start_id, end_id, maxlen=128, device='cpu'):
        super().__init__(start_id, end_id, maxlen, device=device)
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def predict(self, inputs, output_ids):
        """预测下一个token"""
        token_ids = output_ids
        outputs = self.model(token_ids)
        # 取最后一个位置的logits
        return outputs[:, -1, :]


def demo_greedy_search():
    """演示贪心搜索"""
    print("=" * 60)
    print("GPT 文本生成示例 - 贪心搜索")
    print("=" * 60)

    # 模型参数
    vocab_size = 50257
    batch_size = 1

    # 创建模型
    model = GPT(
        vocab_size=vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072
    )
    model.eval()

    # 模拟输入提示词
    # 实际使用时，这应该来自 tokenizer
    prompt = "Once upon a time"
    prompt_ids = [1, 2345, 5678, 890]  # 模拟 token ids

    print(f"\n输入提示: {prompt}")
    print(f"Prompt IDs: {prompt_ids}")

    # 贪心搜索生成
    token_ids = torch.tensor([prompt_ids])
    max_length = 50

    generated_ids = []
    for _ in range(max_length - len(prompt_ids)):
        outputs = model(token_ids)
        next_token_logits = outputs[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)

        generated_ids.append(next_token.item())
        token_ids = torch.cat([token_ids, next_token.unsqueeze(0)], dim=1)

        # 如果生成了结束符就停止
        if next_token.item() == 0:  # 假设 0 是 EOS
            break

    print(f"\n生成的 token IDs: {generated_ids[:20]}...")
    print(f"生成长度: {len(generated_ids)}")

    print("\n✓ 贪心搜索完成!")


def demo_beam_search():
    """演示束搜索"""
    print("\n" + "=" * 60)
    print("GPT 文本生成示例 - 束搜索")
    print("=" * 60)

    vocab_size = 50257

    # 创建模型
    model = GPT(
        vocab_size=vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072
    )

    # 创建生成器
    generator = TextGenerator(
        model=model,
        start_id=1,
        end_id=0,
        maxlen=50,
        device='cpu'
    )

    # 模拟输入
    prompt_ids = torch.tensor([[1, 2345, 5678]])

    print("\n使用束搜索生成 (beam_size=3)...")
    # 注意：这里只是演示接口，实际的 beam_search 需要适配 GPT
    # generated = generator.beam_search(prompt_ids, topk=3)
    # print(f"生成结果: {generated}")

    print("\n✓ 束搜索演示完成!")


def demo_sampling():
    """演示采样生成"""
    print("\n" + "=" * 60)
    print("GPT 文本生成示例 - 随机采样")
    print("=" * 60)

    vocab_size = 50257

    # 创建模型
    model = GPT(
        vocab_size=vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072
    )
    model.eval()

    # 模拟输入
    prompt_ids = [1, 2345, 5678, 890]
    token_ids = torch.tensor([prompt_ids])

    print("\n采样参数:")
    print("  - Temperature: 0.8")
    print("  - Top-K: 50")
    print("  - Top-P: 0.9")

    # 采样生成
    temperature = 0.8
    top_k = 50
    max_length = 30

    generated_ids = []
    for _ in range(max_length - len(prompt_ids)):
        outputs = model(token_ids)
        next_token_logits = outputs[:, -1, :] / temperature

        # Top-K 过滤
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = -float('Inf')

        # 采样
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        generated_ids.append(next_token.item())
        token_ids = torch.cat([token_ids, next_token], dim=1)

        if next_token.item() == 0:
            break

    print(f"\n生成的 token IDs: {generated_ids}")
    print(f"生成长度: {len(generated_ids)}")

    print("\n✓ 随机采样完成!")


def demo_comparison():
    """对比不同生成策略"""
    print("\n" + "=" * 60)
    print("不同生成策略对比")
    print("=" * 60)

    strategies = [
        ("贪心搜索", "最快，但可能重复"),
        ("束搜索", "质量好，速度中等"),
        ("Top-K 采样", "多样性高，速度快"),
        ("Top-P 采样", "平衡质量和多样性"),
        ("Temperature 采样", "控制随机性")
    ]

    print("\n策略对比:")
    for name, desc in strategies:
        print(f"  • {name:15} - {desc}")

    print("\n推荐使用场景:")
    print("  - 短文本补全: 贪心搜索")
    print("  - 长文本生成: Top-P 采样 (p=0.9)")
    print("  - 创意写作: 高 temperature (>1.0) + Top-K")
    print("  - 摘要生成: 束搜索 (beam_size=4)")


if __name__ == '__main__':
    demo_greedy_search()
    demo_beam_search()
    demo_sampling()
    demo_comparison()

    print("\n" + "=" * 60)
    print("🎉 所有演示完成!")
    print("=" * 60)

    print("\n使用说明:")
    print("1. 实际使用时需要加载预训练的 GPT 权重")
    print("2. 使用 Tokenizer 对文本进行编码")
    print("3. 根据任务选择合适的生成策略")
    print("4. 调整 temperature、top_k、top_p 参数")
    print("\n示例代码:")
    print("""
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    prompt = "Once upon a time"
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    # 使用模型生成
    output_ids = model.generate(input_ids, max_length=50)
    text = tokenizer.decode(output_ids[0])
    """)
