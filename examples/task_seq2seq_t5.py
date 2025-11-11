"""Seq2Seq 示例 - T5 自动标题生成"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '..')

from bert4torch.models import T5
from bert4torch.snippets import sequence_padding, AutoRegressiveDecoder


class T5Seq2Seq(nn.Module):
    """T5 Seq2Seq 模型"""
    def __init__(self, vocab_size):
        super().__init__()
        self.t5 = T5(
            vocab_size=vocab_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072
        )

    def forward(self, input_ids, decoder_input_ids):
        return self.t5(input_ids, decoder_input_ids)


class T5Generator(AutoRegressiveDecoder):
    """T5 生成器"""
    def __init__(self, model, tokenizer, start_id, end_id, maxlen=64, device='cpu'):
        super().__init__(start_id, end_id, maxlen, device=device)
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    @torch.no_grad()
    def predict(self, inputs, output_ids):
        """预测下一个token"""
        encoder_outputs = self.model.t5.encode(inputs)
        decoder_outputs = self.model.t5.decode(output_ids, encoder_outputs)
        logits = self.model.t5.lm_head(decoder_outputs)
        return logits[:, -1, :]


def demo_title_generation():
    """演示标题生成"""
    print("=" * 60)
    print("T5 Seq2Seq 示例 - 自动标题生成")
    print("=" * 60)

    # 模型参数
    vocab_size = 32128

    # 创建模型
    model = T5Seq2Seq(vocab_size)
    model.eval()

    # 模拟数据
    # 实际使用时，这些应该来自 tokenizer
    # 输入：新闻正文
    # 输出：新闻标题
    articles = [
        "今天天气很好，阳光明媚，适合出门游玩。",
        "人工智能技术在近年来取得了巨大进展，深度学习成为热门研究方向。",
        "新能源汽车销量持续增长，市场前景广阔。"
    ]

    # 模拟 token ids
    input_ids = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [1, 11, 12, 13, 14, 15, 16, 17, 18],
        [1, 19, 20, 21, 22, 23, 24, 25]
    ]

    print("\n示例数据:")
    for i, article in enumerate(articles):
        print(f"\n[{i+1}] 原文: {article[:30]}...")

    # Padding
    input_ids = sequence_padding(input_ids)
    input_ids = torch.tensor(input_ids)

    print(f"\n输入shape: {input_ids.shape}")

    # 模拟生成（实际需要实现完整的解码逻辑）
    decoder_input_ids = torch.ones((input_ids.shape[0], 1), dtype=torch.long)  # start token

    with torch.no_grad():
        outputs = model(input_ids, decoder_input_ids)

    print(f"输出shape: {outputs.shape}")

    # 模拟生成的标题
    generated_titles = [
        "天气晴好适合出游",
        "人工智能技术发展迅速",
        "新能源汽车市场前景好"
    ]

    print("\n生成的标题:")
    for i, title in enumerate(generated_titles):
        print(f"  [{i+1}] {title}")

    print("\n✓ 标题生成演示完成!")


def demo_translation():
    """演示机器翻译"""
    print("\n" + "=" * 60)
    print("T5 Seq2Seq 示例 - 机器翻译")
    print("=" * 60)

    vocab_size = 32128

    # 创建模型
    model = T5Seq2Seq(vocab_size)
    model.eval()

    # 翻译任务示例
    examples = [
        {
            "source": "translate English to Chinese: Hello, how are you?",
            "target": "你好，你怎么样？"
        },
        {
            "source": "translate English to Chinese: I love machine learning.",
            "target": "我喜欢机器学习。"
        }
    ]

    print("\n翻译示例:")
    for i, example in enumerate(examples):
        print(f"\n[{i+1}]")
        print(f"  输入: {example['source']}")
        print(f"  输出: {example['target']}")

    print("\n说明:")
    print("  T5 使用统一的 text-to-text 格式")
    print("  通过添加前缀来指定任务类型")
    print("  支持多种 NLP 任务（翻译、摘要、问答等）")

    print("\n✓ 机器翻译演示完成!")


def demo_summarization():
    """演示文本摘要"""
    print("\n" + "=" * 60)
    print("T5 Seq2Seq 示例 - 文本摘要")
    print("=" * 60)

    vocab_size = 32128
    model = T5Seq2Seq(vocab_size)
    model.eval()

    # 摘要任务示例
    examples = [
        {
            "document": "summarize: 人工智能是计算机科学的一个分支，它企图了解智能的实质，"
                       "并生产出一种新的能以人类智能相似的方式做出反应的智能机器。"
                       "该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。",
            "summary": "人工智能是计算机科学分支，研究智能机器。"
        }
    ]

    print("\n摘要示例:")
    for i, example in enumerate(examples):
        print(f"\n[{i+1}]")
        print(f"  原文: {example['document'][:50]}...")
        print(f"  摘要: {example['summary']}")

    print("\n摘要策略:")
    print("  - 抽取式: 选择原文中的关键句子")
    print("  - 生成式: 生成新的概括性句子")
    print("  - T5 属于生成式摘要模型")

    print("\n✓ 文本摘要演示完成!")


def demo_training_pipeline():
    """演示训练流程"""
    print("\n" + "=" * 60)
    print("T5 训练流程示例")
    print("=" * 60)

    print("\n训练步骤:")
    print("""
1. 数据准备
   - 收集平行语料（源文本 - 目标文本对）
   - 添加任务前缀（如 "summarize:", "translate:"）
   - 分词和编码

2. 模型训练
   - 输入：源文本的 token ids
   - 输出：目标文本的 token ids
   - 损失函数：交叉熵损失

3. 生成策略
   - Greedy Search: 每步选择概率最高的词
   - Beam Search: 保留多个候选序列
   - Sampling: 根据概率分布采样

4. 评估指标
   - BLEU: 机器翻译
   - ROUGE: 文本摘要
   - Perplexity: 语言模型困惑度
    """)

    print("\n示例代码:")
    print("""
# 训练
for epoch in range(num_epochs):
    for batch in dataloader:
        input_ids, decoder_input_ids, labels = batch

        # 前向传播
        logits = model(input_ids, decoder_input_ids)

        # 计算损失
        loss = criterion(logits.view(-1, vocab_size), labels.view(-1))

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# 生成
input_text = "summarize: ..."
input_ids = tokenizer.encode(input_text)

# 自回归生成
decoder_input_ids = [start_token]
for _ in range(max_length):
    logits = model(input_ids, decoder_input_ids)
    next_token = torch.argmax(logits[:, -1, :])
    decoder_input_ids.append(next_token)
    if next_token == end_token:
        break
    """)


def demo_task_prefixes():
    """演示 T5 的任务前缀"""
    print("\n" + "=" * 60)
    print("T5 任务前缀示例")
    print("=" * 60)

    tasks = [
        ("翻译", "translate English to Chinese: Hello"),
        ("摘要", "summarize: 这是一篇很长的文章..."),
        ("问答", "question: What is AI? context: AI is..."),
        ("分类", "cola sentence: This sentence is grammatical."),
        ("相似度", "stsb sentence1: I like cats. sentence2: I love dogs."),
        ("蕴含", "mnli premise: ... hypothesis: ..."),
    ]

    print("\nT5 支持的任务类型:")
    for task_name, example in tasks:
        print(f"\n  {task_name}:")
        print(f"    {example}")

    print("\n特点:")
    print("  • 统一的 text-to-text 框架")
    print("  • 通过前缀区分不同任务")
    print("  • 可以轻松扩展到新任务")
    print("  • 多任务学习")


if __name__ == '__main__':
    demo_title_generation()
    demo_translation()
    demo_summarization()
    demo_training_pipeline()
    demo_task_prefixes()

    print("\n" + "=" * 60)
    print("🎉 所有演示完成!")
    print("=" * 60)

    print("\n总结:")
    print("  • T5 是强大的 Seq2Seq 模型")
    print("  • 支持多种 NLP 任务")
    print("  • 使用 text-to-text 统一格式")
    print("  • 需要大量平行语料训练")
