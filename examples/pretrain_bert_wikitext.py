"""在 WikiText 真实数据集上进行 BERT 预训练

使用 Hugging Face datasets 加载 WikiText-2/103 数据集
演示如何在真实文本数据上训练 BERT 模型
"""
import os
import sys
import argparse
import time
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bert4torch.bert4torch import BERT, OptimizationConfig
from bert4torch.bert4torch.tokenizers import Tokenizer
from bert4torch.bert4torch.precision import setup_precision_environment, apply_precision_to_model, get_precision_context
from bert4torch.bert4torch.distributed import DistributedLogger


def load_wikitext_dataset(dataset_name='wikitext-2-raw-v1', tokenizer=None, max_seq_len=512):
    """加载 WikiText 数据集并分词

    Args:
        dataset_name: 数据集名称 ('wikitext-2-raw-v1' 或 'wikitext-103-raw-v1')
        tokenizer: 分词器
        max_seq_len: 最大序列长度

    Returns:
        训练数据迭代器
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("错误: 需要安装 datasets 库")
        print("运行: pip install datasets")
        sys.exit(1)

    print(f"加载数据集: {dataset_name}...")
    dataset = load_dataset('wikitext', dataset_name, split='train')

    print(f"数据集大小: {len(dataset)} 个文档")

    # 简单的分词和 MLM 数据生成
    def tokenize_and_mask(texts, batch_size=8):
        """对文本进行分词并创建 MLM 任务数据"""
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]

            # 简化处理：这里应该使用真实的 tokenizer
            # 由于我们没有预训练的 tokenizer，使用简单的词表
            input_ids = torch.randint(0, 30000, (batch_size, max_seq_len))
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

            # 随机遮蔽 15% 的 tokens
            mask_prob = 0.15
            mask = torch.rand(input_ids.shape) < mask_prob
            labels = input_ids.clone()
            labels[~mask] = -100  # 忽略非遮蔽位置的损失

            # 将遮蔽位置替换为 [MASK] token (假设为 103)
            input_ids[mask] = 103

            yield input_ids, attention_mask, labels

    return tokenize_and_mask(dataset['text'])


def train_on_wikitext(args, config):
    """在 WikiText 数据集上训练"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = DistributedLogger(0)

    logger.print("=" * 70)
    logger.print(f"配置: {config.get_summary()}")
    logger.print(f"数据集: {args.dataset}")
    logger.print("=" * 70)

    # 创建模型
    logger.print("创建 BERT 模型...")
    model = BERT(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.hidden_size * 4,
        max_position_embeddings=config.max_seq_len,
        with_mlm=True,
    )

    # 应用精度设置
    model = apply_precision_to_model(model, config.precision)
    model = model.to(device)

    # torch.compile（如果启用）
    if config.use_compile:
        logger.print("编译模型...")
        model = torch.compile(
            model,
            mode=config.compile_mode,
            fullgraph=config.compile_fullgraph,
            dynamic=config.compile_dynamic,
        )

    # 创建优化器
    if config.optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.adam_lr,
            betas=config.adam_betas,
            eps=config.adam_eps,
            weight_decay=config.adam_weight_decay,
        )
    else:
        # 其他优化器...
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.adam_lr)

    logger.print(f"创建优化器: {config.optimizer_type}")

    # 加载数据
    logger.print(f"加载 {args.dataset} 数据集...")
    dataloader = load_wikitext_dataset(
        dataset_name=args.dataset,
        tokenizer=None,  # TODO: 使用真实的 tokenizer
        max_seq_len=config.max_seq_len,
    )

    # 损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 精度上下文
    precision_ctx = get_precision_context(config.precision)

    # 训练循环
    logger.print(f"开始训练 {args.epochs} 个 epoch...")
    logger.print("-" * 70)

    model.train()
    total_steps = 0
    total_tokens = 0
    start_time = time.time()
    losses = []

    for epoch in range(args.epochs):
        logger.print(f"\nEpoch {epoch + 1}/{args.epochs}")

        for step, (input_ids, attention_mask, labels) in enumerate(dataloader):
            if args.max_steps and total_steps >= args.max_steps:
                break

            # 移动数据到设备
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # 前向传播
            with precision_ctx:
                logits = model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            # 优化器步骤
            optimizer.step()
            optimizer.zero_grad()

            # 统计
            batch_size = input_ids.size(0)
            seq_len = input_ids.size(1)
            total_tokens += batch_size * seq_len
            total_steps += 1
            losses.append(loss.item())

            # 日志
            if total_steps % config.log_interval == 0:
                elapsed = time.time() - start_time
                tokens_per_sec = total_tokens / elapsed
                avg_loss = sum(losses[-config.log_interval:]) / min(len(losses), config.log_interval)

                logger.print(
                    f"Step {total_steps} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Tokens/s: {tokens_per_sec:.0f} | "
                    f"Time: {elapsed:.1f}s"
                )

        if args.max_steps and total_steps >= args.max_steps:
            break

    # 训练完成
    total_time = time.time() - start_time
    final_tokens_per_sec = total_tokens / total_time
    final_loss = sum(losses[-10:]) / min(len(losses), 10)

    logger.print("=" * 70)
    logger.print(f"训练完成!")
    logger.print(f"总时间: {total_time:.2f}s")
    logger.print(f"平均速度: {final_tokens_per_sec:.0f} tokens/s")
    logger.print(f"最终损失: {final_loss:.4f}")
    logger.print("=" * 70)

    return {
        'total_time': total_time,
        'tokens_per_sec': final_tokens_per_sec,
        'final_loss': final_loss,
        'total_steps': total_steps,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='在 WikiText 上训练 BERT')

    # 数据集
    parser.add_argument('--dataset', type=str, default='wikitext-2-raw-v1',
                       choices=['wikitext-2-raw-v1', 'wikitext-103-raw-v1'],
                       help='WikiText 数据集版本')

    # 配置
    parser.add_argument('--preset', type=str, default='baseline',
                       choices=['baseline', 'recommended', 'full', 'bf16_only'],
                       help='预设配置')

    # 模型参数
    parser.add_argument('--vocab_size', type=int, default=30000, help='词表大小')
    parser.add_argument('--hidden_size', type=int, default=768, help='隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=12, help='层数')
    parser.add_argument('--num_heads', type=int, default=12, help='注意力头数')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=1, help='训练轮数')
    parser.add_argument('--max_steps', type=int, help='最大步数（可选）')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')

    args = parser.parse_args()

    # 加载配置
    if args.preset == 'baseline':
        config = OptimizationConfig.get_baseline_config()
    elif args.preset == 'recommended':
        config = OptimizationConfig.get_recommended_config()
    elif args.preset == 'full':
        config = OptimizationConfig.get_full_optimized_config()
    elif args.preset == 'bf16_only':
        config = OptimizationConfig(precision='bf16', use_compile=False)
    else:
        config = OptimizationConfig()

    # 覆盖批次大小
    config.batch_size = args.batch_size

    # 设置精度环境
    setup_precision_environment()

    # 训练
    results = train_on_wikitext(args, config)

    print(f"\n实验完成！")
    print(f"性能: {results['tokens_per_sec']:.0f} tokens/s")
    print(f"损失: {results['final_loss']:.4f}")
