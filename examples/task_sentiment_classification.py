"""文本分类示例 - 情感分析"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../bert4torch')

from models import BERT
from snippets import sequence_padding


class SentimentClassifier(nn.Module):
    """BERT文本分类器"""
    def __init__(self, vocab_size, num_labels=2):
        super().__init__()
        self.bert = BERT(
            vocab_size=vocab_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            with_pool=True
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, token_ids, segment_ids):
        hidden, pooled = self.bert(token_ids, segment_ids)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


def demo():
    """演示"""
    print("=" * 60)
    print("BERT 文本分类示例")
    print("=" * 60)

    # 模型参数
    vocab_size = 21128
    num_labels = 2
    batch_size = 4

    # 创建模型
    model = SentimentClassifier(vocab_size, num_labels)
    model.eval()

    # 模拟数据
    # 实际使用时，这些应该来自 tokenizer
    token_ids = [
        [101, 2769, 1599, 6380, 6821, 758, 1429, 102],  # [CLS] 我 喜 欢 这 个 产 品 [SEP]
        [101, 6821, 4495, 1745, 3844, 7770, 102],       # [CLS] 这 电 影 太 烂 [SEP]
        [101, 7555, 2428, 4008, 2658, 102],             # [CLS] 质 量 很 不 错 [SEP]
        [101, 1144, 2636, 1963, 7270, 102],             # [CLS] 太 贵 了 吧 [SEP]
    ]

    # padding
    token_ids = sequence_padding(token_ids)
    segment_ids = torch.zeros_like(torch.tensor(token_ids))
    token_ids = torch.tensor(token_ids)

    print(f"\n输入:")
    print(f"  Token IDs shape: {token_ids.shape}")
    print(f"  Segment IDs shape: {segment_ids.shape}")

    # 前向传播
    with torch.no_grad():
        logits = model(token_ids, segment_ids)

    # 预测
    preds = torch.argmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)

    print(f"\n输出:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Predictions: {preds.tolist()}")
    print(f"\n详细结果:")
    for i in range(batch_size):
        label = "正面" if preds[i] == 1 else "负面"
        confidence = probs[i, preds[i]].item() * 100
        print(f"  样本 {i+1}: {label} (置信度: {confidence:.2f}%)")

    print("\n" + "=" * 60)
    print("✓ 演示完成!")
    print("=" * 60)

    print("\n使用说明:")
    print("1. 实际使用时需要加载预训练的 BERT 权重")
    print("2. 使用 Tokenizer 对文本进行分词和编码")
    print("3. 在标注数据上进行微调")
    print("4. 使用训练好的模型进行预测")


if __name__ == '__main__':
    demo()
