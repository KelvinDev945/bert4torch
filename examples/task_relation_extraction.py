"""关系抽取示例 - BERT + GlobalPointer"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '..')

from bert4torch.models import BERT
from bert4torch.layers import GlobalPointer
from bert4torch.snippets import sequence_padding


class RelationExtractionModel(nn.Module):
    """关系抽取模型（BERT + GlobalPointer）"""
    def __init__(self, vocab_size, num_relations, head_size=64):
        super().__init__()
        self.bert = BERT(
            vocab_size=vocab_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072
        )
        self.dropout = nn.Dropout(0.1)
        # GlobalPointer for relation extraction
        # heads = num_relations (每种关系一个头)
        self.global_pointer = GlobalPointer(
            hidden_size=768,
            heads=num_relations,
            head_size=head_size,
            use_rope=True
        )

    def forward(self, token_ids, segment_ids):
        hidden = self.bert(token_ids, segment_ids)
        hidden = self.dropout(hidden)
        # scores: [batch, num_relations, seq_len, seq_len]
        scores = self.global_pointer(hidden)
        return scores


def demo_entity_pair_extraction():
    """演示实体对抽取"""
    print("=" * 60)
    print("关系抽取示例 - 实体对抽取")
    print("=" * 60)

    # 参数
    vocab_size = 21128
    # 关系类型：人物-出生地、人物-职业、公司-创始人等
    relations = ['PER-BIRTH', 'PER-JOB', 'COM-FOUNDER', 'COM-LOC']
    num_relations = len(relations)

    # 创建模型
    model = RelationExtractionModel(vocab_size, num_relations)
    model.eval()

    # 模拟数据
    # 文本: "马云创建了阿里巴巴公司，总部位于杭州"
    # 实体: 马云(PER), 阿里巴巴(COM), 杭州(LOC)
    # 关系: (马云, 创建, 阿里巴巴), (阿里巴巴, 位于, 杭州)

    texts = [
        "马云创建了阿里巴巴公司，总部位于杭州",
        "比尔盖茨是微软公司的创始人"
    ]

    # 模拟 token ids
    token_ids = [
        [101, 7722, 756, 1140, 2456, 749, 7350, 7030, 7030, 1086, 1385, 102],
        [101, 3683, 2209, 4919, 4868, 3221, 2523, 6763, 1086, 1385, 102]
    ]

    print("\n输入文本:")
    for i, text in enumerate(texts):
        print(f"  [{i+1}] {text}")

    # Padding
    token_ids = sequence_padding(token_ids)
    segment_ids = torch.zeros_like(torch.tensor(token_ids))
    token_ids = torch.tensor(token_ids)

    print(f"\nToken IDs shape: {token_ids.shape}")

    # 前向传播
    with torch.no_grad():
        scores = model(token_ids, segment_ids)

    print(f"输出 scores shape: {scores.shape}")
    print(f"  [batch_size, num_relations, seq_len, seq_len]")

    # 模拟解码结果
    print("\n提取的关系:")
    relations_extracted = [
        ("马云", "COM-FOUNDER", "阿里巴巴", 0.95),
        ("阿里巴巴", "COM-LOC", "杭州", 0.88),
    ]

    for subj, rel, obj, score in relations_extracted:
        print(f"  ({subj}, {rel}, {obj}) - 置信度: {score:.2f}")

    print("\n✓ 实体对抽取完成!")


def demo_multilabel_classification():
    """演示多标签分类"""
    print("\n" + "=" * 60)
    print("关系抽取示例 - 多标签分类方式")
    print("=" * 60)

    print("\nGlobalPointer 工作原理:")
    print("""
1. 输入文本经过 BERT 编码
2. GlobalPointer 计算每对位置的关系得分
3. 对于每种关系类型，得到一个 [seq_len, seq_len] 的矩阵
4. 矩阵中 (i, j) 表示从位置 i 到位置 j 存在该关系的得分
5. 使用阈值或 Top-K 选择最可能的关系三元组
    """)

    print("\n优点:")
    print("  • 可以处理实体重叠问题")
    print("  • 可以一次抽取多个关系")
    print("  • 使用 RoPE 位置编码增强位置信息")
    print("  • 训练和推理效率高")

    print("\n与传统方法对比:")
    print("  传统方法: 先识别实体，再分类关系")
    print("  GlobalPointer: 端到端直接抽取关系")


def demo_training():
    """演示训练过程"""
    print("\n" + "=" * 60)
    print("关系抽取模型训练")
    print("=" * 60)

    print("\n数据标注格式:")
    print("""
{
    "text": "马云创建了阿里巴巴公司",
    "relations": [
        {
            "subject": "马云",
            "subject_type": "PER",
            "predicate": "创建",
            "object": "阿里巴巴",
            "object_type": "COM",
            "relation_type": "COM-FOUNDER"
        }
    ]
}
    """)

    print("\n损失函数:")
    print("""
def multilabel_categorical_crossentropy(y_true, y_pred):
    # y_true: [batch, heads, seq_len, seq_len]
    # y_pred: [batch, heads, seq_len, seq_len]

    # 对于正样本，最大化预测分数
    # 对于负样本，最小化预测分数

    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    y_pred_neg = y_pred - y_true * 1e12

    y_pred_pos = torch.logsumexp(y_pred_pos, dim=-1)
    y_pred_neg = torch.logsumexp(y_pred_neg, dim=-1)

    loss = torch.logsumexp(torch.stack([y_pred_pos, y_pred_neg], dim=-1), dim=-1)
    return loss.mean()
    """)

    print("\n训练技巧:")
    print("  • 使用对抗训练提升鲁棒性")
    print("  • 数据增强（同义词替换、回译等）")
    print("  • 负采样策略")
    print("  • 类别平衡")


def demo_inference():
    """演示推理过程"""
    print("\n" + "=" * 60)
    print("关系抽取推理")
    print("=" * 60)

    print("\n推理步骤:")
    print("""
1. 文本编码
   text = "马云创建了阿里巴巴公司"
   token_ids = tokenizer.encode(text)

2. 模型预测
   scores = model(token_ids)  # [1, num_relations, seq_len, seq_len]

3. 解码
   for relation_id in range(num_relations):
       relation_scores = scores[0, relation_id]  # [seq_len, seq_len]

       # 找出得分 > 阈值的位置对
       indices = torch.where(relation_scores > threshold)

       for i, j in zip(indices[0], indices[1]):
           subject = text[i:j+1]
           object = text[...]
           relation = relation_names[relation_id]
           print(f"({subject}, {relation}, {object})")
    """)

    print("\n后处理:")
    print("  • 去除重叠的实体对")
    print("  • 合并相似的关系")
    print("  • 规则过滤（如长度、类型匹配）")


def demo_use_cases():
    """演示应用场景"""
    print("\n" + "=" * 60)
    print("关系抽取应用场景")
    print("=" * 60)

    use_cases = [
        ("知识图谱构建", "从文本中抽取实体和关系，构建知识图谱"),
        ("信息抽取", "从新闻、论文中抽取结构化信息"),
        ("事件抽取", "识别事件的参与者、时间、地点等要素"),
        ("医疗领域", "抽取疾病-症状、药物-副作用等关系"),
        ("金融领域", "抽取公司-高管、投资-被投等关系"),
    ]

    print("\n应用场景:")
    for name, desc in use_cases:
        print(f"\n  {name}:")
        print(f"    {desc}")

    print("\n数据集:")
    print("  • DuIE: 百度关系抽取数据集")
    print("  • NYT: 纽约时报关系抽取数据集")
    print("  • WebNLG: 知识图谱到文本数据集")
    print("  • SemEval: 语义评测关系抽取任务")


if __name__ == '__main__':
    demo_entity_pair_extraction()
    demo_multilabel_classification()
    demo_training()
    demo_inference()
    demo_use_cases()

    print("\n" + "=" * 60)
    print("🎉 所有演示完成!")
    print("=" * 60)

    print("\n总结:")
    print("  • GlobalPointer 是高效的关系抽取方法")
    print("  • 可以处理实体重叠和嵌套")
    print("  • 端到端训练，无需pipeline")
    print("  • 适用于多种信息抽取任务")
