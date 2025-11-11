# bert4torch 开发进度

## ✅ 已完成

- [x] 创建 style_examples 目录和三种风格的 MultiHeadAttention 示例
- [x] 等待用户选择代码风格（选择：风格 A 简洁风格）
- [x] 搭建项目基础结构（目录、__init__.py、setup.py）
- [x] 实现 backend.py（位置编码、mask、激活函数等）
- [x] 实现 layers.py（MultiHeadAttention、FeedForward、LayerNorm 等）
- [x] 实现 models.py 基础（BERT、RoFormer、GPT、T5 模型）
- [x] 实现 optimizers.py（AdamW、学习率调度、EMA、梯度累积等）
- [x] 实现 snippets.py（数据处理、解码器等工具函数）
- [x] 实现 tokenizers.py（BERT分词器）

## 🚧 进行中

- [ ] 编写示例代码验证功能

## 📋 待完成

- [ ] 编写示例代码
  - [ ] 基础示例（特征提取、MLM测试）
  - [ ] 文本分类示例
  - [ ] 序列标注示例（NER + CRF）
  - [ ] 文本生成示例（GPT）
  - [ ] Seq2Seq 示例（T5）
- [ ] 编写单元测试
- [ ] 完善文档

## 📝 开发日志

### 2025-11-11

#### 完成的工作

1. **项目结构搭建**
   - 创建 `bert4torch/` 主包目录
   - 创建 `examples/`、`tests/` 目录
   - 编写 `setup.py` 和 `README.md`

2. **backend.py 实现**
   - `gelu()`: GELU 激活函数
   - `sinusoidal_embeddings()`: 正弦位置编码
   - `apply_rotary_position_embeddings()`: RoPE 旋转位置编码
   - `sequence_masking()`: 序列mask操作
   - `attention_normalize()`: attention归一化
   - `piecewise_linear()`: 分段线性函数（学习率调度）

3. **layers.py 实现**
   - `MultiHeadAttention`: 多头注意力（支持交叉注意力、位置偏置）
   - `FeedForward`: 前馈网络
   - `LayerNorm`: 层归一化（支持条件LN）
   - `Embedding`: 嵌入层
   - `PositionEmbedding`: 可学习位置编码
   - `SinusoidalPositionEmbedding`: 正弦位置编码
   - `RoPEPositionEmbedding`: 旋转位置编码
   - `RelativePositionEmbedding`: T5相对位置编码
   - `GlobalPointer`: 全局指针（实体识别）
   - `CRF`: 条件随机场（序列标注）

4. **models.py 实现**
   - `Transformer`: 基类，定义统一接口
   - `BERT`: 标准BERT实现（支持 MLM、NSP、pooler）
   - `BERTLayer`: BERT Transformer层
   - `RoFormer`: 带RoPE的BERT
   - `RoFormerLayer`: RoFormer层
   - `GPT`: GPT单向语言模型
   - `GPTLayer`: GPT Transformer层
   - `T5`: T5 Encoder-Decoder模型
   - `T5Stack`: T5编码器/解码器栈
   - `T5Layer`: T5 Transformer层
   - `build_transformer_model()`: 统一模型构建接口

#### 代码特点

- 采用**风格 A（bert4keras 简洁风格）**
- 代码简洁，变量命名简短（q, k, v, o）
- 最少的注释和文档
- 单文件组织，便于阅读和修改

5. **optimizers.py 实现**
   - `AdamW`: AdamW 优化器
   - `extend_with_weight_decay()`: 权重衰减装饰器
   - `extend_with_piecewise_linear_lr()`: 分段线性学习率装饰器
   - `extend_with_gradient_accumulation()`: 梯度累积装饰器
   - `extend_with_exponential_moving_average()`: EMA 装饰器
   - `extend_with_lookahead()`: Lookahead 装饰器
   - `get_linear_schedule_with_warmup()`: 线性 warmup 调度器
   - `get_cosine_schedule_with_warmup()`: 余弦 warmup 调度器

6. **snippets.py 实现**
   - `sequence_padding()`: 序列填充
   - `truncate_sequences()`: 序列截断
   - `text_segmentate()`: 文本分段
   - `DataGenerator`: 数据生成器基类
   - `AutoRegressiveDecoder`: 自回归解码器（beam search、random sample）
   - `ViterbiDecoder`: 维特比解码器
   - `parallel_apply()`: 并行处理
   - 装饰器工具函数

7. **tokenizers.py 实现**
   - `TokenizerBase`: 分词器基类
   - `Tokenizer`: BERT 分词器
   - `load_vocab()`: 加载词表
   - `save_vocab()`: 保存词表
   - WordPiece 分词
   - 中文字符处理
   - `rematch()`: token 映射回原文本

#### 下一步计划

1. 编写示例代码验证功能
2. 编写单元测试
3. 完善文档
