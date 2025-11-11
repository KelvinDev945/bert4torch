# bert4torch

bert4keras 的 PyTorch 实现版本，保持简洁易读的代码风格。

## 安装

```bash
pip install -e .
```

## 快速开始

```python
import torch
from bert4torch import build_transformer_model

# 构建 BERT 模型
model = build_transformer_model(
    config_path='path/to/config.json',
    checkpoint_path='path/to/pytorch_model.bin',
    model='bert'
)

# 使用模型
inputs = torch.randint(0, 1000, (2, 128))
outputs = model(inputs)
```

## 特性

- 简洁易读的代码
- 支持 BERT、GPT、T5 等主流模型
- 兼容 TensorFlow checkpoint 加载
- 丰富的示例代码

## 项目结构

```
bert4torch/
├── backend.py      # 工具函数
├── layers.py       # 自定义层
├── models.py       # 模型实现
├── optimizers.py   # 优化器
├── snippets.py     # 辅助工具
└── tokenizers.py   # 分词器
```
