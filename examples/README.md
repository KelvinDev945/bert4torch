# bert4torch 示例代码

本目录包含 bert4torch 的各种使用示例。

## 📚 示例列表

### 基础示例
- **basic_test.py** - 基础功能测试（MultiHeadAttention、BERT、GPT、T5）

### 任务示例
- **task_sentiment_classification.py** - 文本分类（情感分析）
- **task_ner_crf.py** - 序列标注（命名实体识别 + CRF）
- **task_text_generation_gpt.py** - 文本生成（GPT + 多种生成策略）
- **task_seq2seq_t5.py** - Seq2Seq（T5 标题生成/翻译/摘要）
- **task_relation_extraction.py** - 关系抽取（BERT + GlobalPointer）

## 🚀 运行示例

**注意**：由于包结构问题，示例代码目前需要使用特定方式运行。

### 方法 1：使用 Python 模块方式

```bash
# 从项目根目录运行
cd /path/to/bert4torch

# 方式 1: 修改 PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH
python examples/basic_test.py

# 方式 2: 使用 python -m
python -m examples.basic_test
```

### 方法 2：直接查看代码

所有示例都是自包含的，您可以直接查看代码了解使用方法：

```bash
cat examples/basic_test.py
cat examples/task_sentiment_classification.py
# ... 其他示例
```

### 方法 3：复制核心代码

您可以将 `bert4torch/bert4torch/` 目录下的文件复制到您的项目中直接使用。

## 📖 示例说明

### basic_test.py
测试所有核心组件的基本功能：
- MultiHeadAttention 层
- FeedForward 层
- BERT、RoFormer、GPT、T5 模型

### task_sentiment_classification.py
演示如何使用 BERT 进行文本分类：
- 构建分类器
- 模拟数据和前向传播
- 预测和结果展示

### task_ner_crf.py
演示序列标注（NER）：
- BERT + CRF 模型
- BIO 标注方案
- 实体提取

### task_text_generation_gpt.py
演示文本生成：
- 贪心搜索
- 束搜索（Beam Search）
- 随机采样（Top-K / Top-P）
- 温度（Temperature）控制
- 不同策略对比

### task_seq2seq_t5.py
演示 T5 的 Seq2Seq 任务：
- 标题生成
- 机器翻译
- 文本摘要
- 训练流程
- T5 任务前缀

### task_relation_extraction.py
演示关系抽取：
- BERT + GlobalPointer
- 实体对抽取
- 多标签分类
- 训练和推理流程

## 💡 使用建议

1. **先查看代码**：所有示例都包含详细的注释和说明
2. **理解原理**：重点理解模型结构和数据流
3. **适配实际任务**：根据您的需求修改示例代码
4. **加载预训练权重**：实际使用时需要加载预训练模型

## 🔧 常见问题

### Q: 为什么无法直接运行示例？
A: 由于包结构问题（bert4torch/bert4torch/），需要设置 PYTHONPATH 或使用其他方式运行。

### Q: 如何在我的项目中使用？
A: 复制 `bert4torch/bert4torch/` 目录到您的项目，或参考示例代码实现。

### Q: 示例代码可以直接用于生产吗？
A: 示例代码主要用于演示，生产环境需要：
- 加载真实的预训练权重
- 使用真实的 tokenizer
- 完整的训练和评估流程
- 错误处理和日志

## 📝 代码风格

所有示例遵循**风格 A（bert4keras 简洁风格）**：
- 变量名简短（q, k, v, o）
- 代码简洁易读
- 最少的注释
- 专注核心逻辑

## 🤝 贡献

欢迎提交新的示例！请确保：
- 代码简洁清晰
- 包含必要的注释
- 遵循项目代码风格
