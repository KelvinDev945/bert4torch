"""BERT 快速训练示例

使用 Bert4torch + modded-nanogpt 优化技术进行快速训练
支持：BF16、Muon 优化器、torch.compile、分布式训练等
"""
import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from bert4torch.bert4torch import (
    BERT,
    OptimizationConfig,
    Muon, NorMuon, AdamW,
    convert_to_bfloat16,
    setup_precision_environment,
    get_precision_context,
    apply_precision_to_model,
    setup_distributed,
    is_distributed,
    get_rank,
    wrap_model_ddp,
    DistributedLogger,
)


def create_dummy_data(batch_size, seq_len, vocab_size, num_batches=100):
    """创建虚拟数据用于快速测试"""
    for _ in range(num_batches):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        yield input_ids, attention_mask, labels


def train_step(model, batch, criterion, precision_ctx):
    """单步训练"""
    input_ids, attention_mask, labels = batch

    with precision_ctx:
        # 前向传播（BERT模型会自动计算attention_mask）
        logits = model(input_ids)

        # 计算损失（简化的 MLM 损失）
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

    return loss


def main(args):
    # 加载配置
    if args.config:
        config = OptimizationConfig.from_yaml(args.config)
    else:
        # 使用预设配置
        if args.preset == 'baseline':
            config = OptimizationConfig.get_baseline_config()
        elif args.preset == 'recommended':
            config = OptimizationConfig.get_recommended_config()
        elif args.preset == 'full':
            config = OptimizationConfig.get_full_optimized_config()
        elif args.preset == 'single_gpu':
            config = OptimizationConfig.get_single_gpu_config()
        else:
            config = OptimizationConfig()

    # 覆盖命令行参数
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.precision:
        config.precision = args.precision

    # 设置精度环境
    setup_precision_environment()

    # 初始化分布式（如果需要）
    if config.use_distributed:
        rank, world_size, local_rank = setup_distributed(backend=config.ddp_backend)
    else:
        rank, world_size, local_rank = 0, 1, 0

    # 日志记录器
    logger = DistributedLogger(rank)
    logger.print("=" * 70)
    logger.print(f"配置: {config.get_summary()}")
    logger.print("=" * 70)

    # 设备
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

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
            dynamic=config.compile_dynamic,
            fullgraph=config.compile_fullgraph,
            mode=config.compile_mode,
        )

    # DDP 包装（如果需要）
    if config.use_distributed:
        logger.print("包装 DDP...")
        model = wrap_model_ddp(
            model,
            gradient_as_bucket_view=config.gradient_as_bucket_view,
            static_graph=config.static_graph,
        )

    # 创建优化器
    logger.print(f"创建优化器: {config.optimizer_type}")

    # 分离参数组（如果需要）
    if config.optimizer_type in ['muon', 'normuon']:
        # Muon: 仅优化 2D 矩阵参数
        matrix_params = [p for p in model.parameters() if p.ndim >= 2 and p.requires_grad]

        optimizer = NorMuon(
            matrix_params,
            lr=config.muon_lr,
            momentum=config.muon_momentum,
            beta2=config.muon_beta2,
        )
    else:
        # Adam/AdamW
        optimizer = AdamW(
            model.parameters(),
            lr=config.adam_lr,
            betas=config.adam_betas,
            eps=config.adam_eps,
            weight_decay=config.adam_weight_decay,
        )

    # 损失函数
    criterion = nn.CrossEntropyLoss()

    # 精度上下文
    precision_ctx = get_precision_context(config.precision)

    # 训练循环
    logger.print(f"开始训练 {args.max_steps} 步...")
    logger.print("-" * 70)

    model.train()
    total_tokens = 0
    start_time = time.time()
    losses = []

    # 创建数据
    dataloader = create_dummy_data(
        config.batch_size,
        config.max_seq_len,
        args.vocab_size,
        num_batches=args.max_steps,
    )

    for step, batch in enumerate(dataloader):
        if step >= args.max_steps:
            break

        # CUDA Graphs 步骤标记（修复 FP8 + torch.compile 兼容性）
        if config.use_compile and config.fp8_lm_head:
            torch.compiler.cudagraph_mark_step_begin()

        # 移动数据到设备
        batch = [b.to(device) for b in batch]

        # 前向+反向
        loss = train_step(model, batch, criterion, precision_ctx)
        loss.backward()

        # 梯度裁剪
        if config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

        # 优化器步骤
        optimizer.step()
        optimizer.zero_grad()

        # 统计
        total_tokens += config.batch_size * config.max_seq_len
        losses.append(loss.item())

        # 日志
        if (step + 1) % config.log_interval == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed
            avg_loss = sum(losses[-config.log_interval:]) / min(len(losses), config.log_interval)

            logger.print(
                f"Step {step + 1}/{args.max_steps} | "
                f"Loss: {avg_loss:.4f} | "
                f"Tokens/s: {tokens_per_sec:.0f} | "
                f"Time: {elapsed:.1f}s"
            )

    # 最终统计
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
        'config': config.get_summary(),
        'total_time': total_time,
        'tokens_per_sec': final_tokens_per_sec,
        'final_loss': final_loss,
        'total_steps': args.max_steps,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BERT 快速训练')

    # 配置
    parser.add_argument('--config', type=str, help='配置文件路径 (YAML)')
    parser.add_argument('--preset', type=str, default='baseline',
                       choices=['baseline', 'recommended', 'full', 'single_gpu'],
                       help='预设配置')

    # 模型参数
    parser.add_argument('--vocab_size', type=int, default=30000, help='词表大小')
    parser.add_argument('--hidden_size', type=int, default=768, help='隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=12, help='层数')
    parser.add_argument('--num_heads', type=int, default=12, help='注意力头数')

    # 训练参数
    parser.add_argument('--batch_size', type=int, help='批次大小')
    parser.add_argument('--max_steps', type=int, default=500, help='最大步数')
    parser.add_argument('--precision', type=str, choices=['fp32', 'bf16', 'fp16', 'fp8'],
                       help='精度设置')

    args = parser.parse_args()

    # 运行训练
    results = main(args)

    # 保存结果
    import json
    results_file = 'training_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存到 {results_file}")
