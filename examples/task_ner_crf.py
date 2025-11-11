"""序列标注示例 - 命名实体识别 (BERT + CRF)"""
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '..')
sys.path.insert(0, '../bert4torch')

from models import BERT
from layers import CRF
from snippets import sequence_padding


class NERModel(nn.Module):
    """BERT + CRF 命名实体识别模型"""
    def __init__(self, vocab_size, num_tags):
        super().__init__()
        self.bert = BERT(
            vocab_size=vocab_size,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072
        )
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, num_tags)
        self.crf = CRF(num_tags)

    def forward(self, token_ids, segment_ids, tags=None):
        hidden = self.bert(token_ids, segment_ids)
        hidden = self.dropout(hidden)
        emissions = self.classifier(hidden)

        if tags is not None:
            # 训练模式：计算loss
            mask = (token_ids != 0)
            loss = self.crf(emissions, tags, mask)
            return loss
        else:
            # 预测模式：解码
            mask = (token_ids != 0)
            preds = self.crf.decode(emissions, mask)
            return preds


def demo():
    """演示"""
    print("=" * 60)
    print("BERT + CRF 命名实体识别示例")
    print("=" * 60)

    # 参数
    vocab_size = 21128
    # BIO标注: O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG
    tag_names = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG']
    num_tags = len(tag_names)

    # 创建模型
    model = NERModel(vocab_size, num_tags)
    model.eval()

    # 模拟数据
    # 实际使用时，这些应该来自 tokenizer
    # "张三去北京工作" -> [CLS] 张 三 去 北 京 工 作 [SEP]
    token_ids = [
        [101, 2476, 676, 1343, 1266, 776, 2339, 868, 102],
    ]

    # padding
    token_ids = sequence_padding(token_ids)
    segment_ids = torch.zeros_like(torch.tensor(token_ids))
    token_ids = torch.tensor(token_ids)

    print(f"\n输入:")
    print(f"  Token IDs shape: {token_ids.shape}")
    print(f"  Token IDs: {token_ids[0].tolist()}")

    # 预测
    with torch.no_grad():
        preds = model(token_ids, segment_ids)

    print(f"\n输出:")
    print(f"  预测标签 (索引): {preds[0].tolist()}")

    # 转换为标签名
    pred_tags = [tag_names[i] for i in preds[0].tolist()]
    print(f"  预测标签 (名称): {pred_tags}")

    # 提取实体
    print(f"\n提取的实体:")
    entities = []
    entity = []
    entity_type = None

    for i, tag in enumerate(pred_tags):
        if tag.startswith('B-'):
            if entity:
                entities.append((entity_type, entity))
            entity = [i]
            entity_type = tag[2:]
        elif tag.startswith('I-') and entity:
            entity.append(i)
        else:
            if entity:
                entities.append((entity_type, entity))
                entity = []
                entity_type = None

    if entity:
        entities.append((entity_type, entity))

    for ent_type, positions in entities:
        print(f"  类型: {ent_type}, 位置: {positions}")

    print("\n" + "=" * 60)
    print("✓ 演示完成!")
    print("=" * 60)

    print("\n模型说明:")
    print("• BERT 提取特征")
    print("• Linear 层输出每个token的标签分数")
    print("• CRF 层进行全局解码，确保标签序列合法")
    print("• 训练时使用 CRF loss")
    print("• 预测时使用维特比解码")


if __name__ == '__main__':
    demo()
