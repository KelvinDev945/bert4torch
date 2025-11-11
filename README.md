# bert4torch

bert4keras 的 PyTorch 实现版本，保持简洁易读的代码风格。

## ✨ 特性

- 🚀 **简洁易读**：采用 bert4keras 简洁风格（风格 A），代码清晰易懂
- 🔧 **完整功能**：支持 BERT、RoFormer、GPT、T5 等主流模型
- 🎯 **开箱即用**：提供丰富的示例代码
- ⚡ **高性能**：支持多种优化技巧（EMA、梯度累积、混合精度等）
- 🛠️ **易于扩展**：模块化设计，方便添加新功能

## 📦 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/bert4torch.git
cd bert4torch

# 直接使用（不需要安装）
cd examples
python basic_test.py
```

## 🚀 快速开始

```python
import torch
import sys
sys.path.insert(0, '..')

from bert4torch.models import BERT

# 创建 BERT 模型
model = BERT(
    vocab_size=21128,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072
)

# 输入数据
token_ids = torch.randint(0, 21128, (2, 128))
segment_ids = torch.zeros(2, 128, dtype=torch.long)

# 前向传播
output = model(token_ids, segment_ids)
print(f"Output shape: {output.shape}")  # [2, 128, 768]
```

## 📚 示例代码

```bash
cd examples

# 基础功能测试
python basic_test.py

# 文本分类
python task_sentiment_classification.py

# 命名实体识别 (BERT + CRF)
python task_ner_crf.py
```

## 🏗️ 项目结构

```
bert4torch/
├── bert4torch/bert4torch/   # 核心库代码
│   ├── backend.py           # 工具函数
│   ├── layers.py            # 自定义层
│   ├── models.py            # 模型实现
│   ├── optimizers.py        # 优化器
│   ├── snippets.py          # 辅助工具
│   └── tokenizers.py        # 分词器
├── examples/                # 示例代码
├── style_examples/          # 代码风格示例
└── README.md
```

## 🔑 核心功能

### 支持的模型

- **BERT**：标准 BERT 模型
- **RoFormer**：带 RoPE 的 BERT
- **GPT**：单向语言模型
- **T5**：Encoder-Decoder 模型

### 主要组件

- MultiHeadAttention、FeedForward、LayerNorm
- CRF（条件随机场）
- GlobalPointer（实体识别）
- AdamW 优化器 + 多种训练技巧
- AutoRegressiveDecoder（文本生成）

## 💡 设计特点

- **风格 A**：简洁风格，变量名简短（q/k/v/o）
- **易扩展**：模块化设计
- **全功能**：完整实现 bert4keras 核心功能

## 📄 许可证

Apache License 2.0

## 🙏 致谢

感谢 [bert4keras](https://github.com/bojone/bert4keras) 项目！
