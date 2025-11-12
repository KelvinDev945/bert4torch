"""运行全面的优化组合实验

自动测试15个不同的优化配置组合，对比性能
"""
import os
import sys
import json
import subprocess
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from bert4torch.bert4torch import create_experiment_configs


def run_experiment(config_name, config, max_steps=500):
    """运行单个实验"""
    print(f"\n{'='*70}")
    print(f"实验: {config_name}")
    print(f"配置: {config.get_summary()}")
    print(f"{'='*70}\n")

    # 保存配置到临时文件
    config_file = f'/tmp/config_{config_name}.yaml'
    config.to_yaml(config_file)

    # 运行训练
    cmd = [
        sys.executable,
        'examples/pretrain_bert_fast.py',
        '--config', config_file,
        '--max_steps', str(max_steps),
    ]

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10分钟超时
        )

        success = result.returncode == 0
        elapsed_time = time.time() - start_time

        # 读取结果
        if success and os.path.exists('training_results.json'):
            with open('training_results.json', 'r') as f:
                results = json.load(f)
        else:
            results = {
                'config': config.get_summary(),
                'error': result.stderr if not success else 'Unknown error',
            }

        results['success'] = success
        results['elapsed_time'] = elapsed_time

    except subprocess.TimeoutExpired:
        results = {
            'config': config.get_summary(),
            'success': False,
            'error': 'Timeout (>10min)',
            'elapsed_time': 600,
        }
    except Exception as e:
        results = {
            'config': config.get_summary(),
            'success': False,
            'error': str(e),
            'elapsed_time': time.time() - start_time,
        }

    finally:
        # 清理临时文件
        if os.path.exists(config_file):
            os.remove(config_file)

    return results


def main():
    # 创建实验目录
    exp_dir = Path('experiments')
    exp_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = exp_dir / f'results_{timestamp}'
    results_dir.mkdir(exist_ok=True)

    print(f"实验结果将保存到: {results_dir}")

    # 获取所有实验配置
    configs = create_experiment_configs()

    print(f"\n将运行 {len(configs)} 个实验配置\n")

    # 运行所有实验
    all_results = {}

    for config_name, config in configs.items():
        results = run_experiment(config_name, config, max_steps=500)
        all_results[config_name] = results

        # 保存单个实验结果
        result_file = results_dir / f'{config_name}.json'
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)

        # 实时显示结果
        if results.get('success'):
            print(f"\n✓ {config_name}: {results.get('tokens_per_sec', 0):.0f} tokens/s")
        else:
            print(f"\n✗ {config_name}: 失败 - {results.get('error', 'Unknown')}")

    # 保存汇总结果
    summary_file = results_dir / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # 生成对比报告
    generate_comparison_report(all_results, results_dir / 'comparison.md')

    print(f"\n{'='*70}")
    print("所有实验完成!")
    print(f"结果保存在: {results_dir}")
    print(f"{'='*70}\n")


def generate_comparison_report(results, output_file):
    """生成对比报告"""
    lines = []
    lines.append("# BERT 快速训练优化对比实验\n")
    lines.append(f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("## 实验结果对比\n")
    lines.append("| 配置 | 速度 (tokens/s) | 损失 | 时间 (s) | 状态 |")
    lines.append("|------|----------------|------|----------|------|")

    # 按速度排序
    sorted_results = sorted(
        results.items(),
        key=lambda x: x[1].get('tokens_per_sec', 0),
        reverse=True
    )

    for config_name, result in sorted_results:
        if result.get('success'):
            speed = result.get('tokens_per_sec', 0)
            loss = result.get('final_loss', 0)
            time_taken = result.get('total_time', 0)
            status = "✓"
        else:
            speed = 0
            loss = '-'
            time_taken = result.get('elapsed_time', 0)
            status = "✗"

        lines.append(
            f"| {config_name} | {speed:.0f} | {loss} | {time_taken:.1f} | {status} |"
        )

    lines.append("\n## 优化建议\n")

    # 找出最快配置
    best_config = max(results.items(), key=lambda x: x[1].get('tokens_per_sec', 0))
    if best_config[1].get('success'):
        lines.append(f"**最快配置**: {best_config[0]}")
        lines.append(f"- 速度: {best_config[1]['tokens_per_sec']:.0f} tokens/s")
        lines.append(f"- 配置摘要: {best_config[1]['config']}\n")

    # 基线对比
    baseline = results.get('01_baseline', {})
    if baseline.get('success'):
        baseline_speed = baseline.get('tokens_per_sec', 1)
        lines.append(f"\n## 相对基线加速比\n")
        lines.append("| 配置 | 加速比 |")
        lines.append("|------|--------|")

        for config_name, result in sorted_results:
            if result.get('success'):
                speedup = result.get('tokens_per_sec', 0) / baseline_speed
                lines.append(f"| {config_name} | {speedup:.2f}x |")

    # 写入文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\n对比报告已生成: {output_file}")


if __name__ == '__main__':
    main()
