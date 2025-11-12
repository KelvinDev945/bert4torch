"""基础功能测试"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from bert4torch.bert4torch import (
    OptimizationConfig,
    Muon, NorMuon,
    polar_express,
    convert_to_bfloat16,
    check_fp8_support,
    BERT,
)


def test_config():
    """测试配置系统"""
    print("测试配置系统...")

    # 基线配置
    config = OptimizationConfig.get_baseline_config()
    assert config.precision == 'fp32'
    assert config.use_compile == False

    # 推荐配置
    config = OptimizationConfig.get_recommended_config()
    assert config.precision == 'bf16'
    assert config.use_compile == True
    assert config.optimizer_type == 'normuon'

    # 配置摘要
    summary = config.get_summary()
    assert 'BF16' in summary
    assert 'Compile' in summary

    print("✓ 配置系统测试通过")


def test_polar_express():
    """测试 Polar Express 正交化"""
    print("测试 Polar Express...")

    # 创建随机矩阵
    x = torch.randn(10, 8)

    # 正交化
    u = polar_express(x)

    # 验证正交性: U^T @ U ≈ I
    uu = u.T @ u
    identity = torch.eye(8)

    error = (uu - identity).abs().max().item()
    assert error < 0.1, f"正交化误差过大: {error}"

    print(f"✓ Polar Express 测试通过 (误差: {error:.6f})")


def test_muon_optimizer():
    """测试 Muon 优化器"""
    print("测试 Muon 优化器...")

    # 创建简单模型
    model = torch.nn.Linear(10, 5)
    optimizer = NorMuon(model.parameters(), lr=0.01)

    # 模拟训练步骤
    for _ in range(5):
        x = torch.randn(32, 10)
        y = torch.randn(32, 5)

        output = model(x)
        loss = ((output - y) ** 2).mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print("✓ Muon 优化器测试通过")


def test_bert_model():
    """测试 BERT 模型"""
    print("测试 BERT 模型...")

    model = BERT(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        with_mlm=True,
    )

    # 前向传播
    input_ids = torch.randint(0, 1000, (4, 32))
    attention_mask = torch.ones(4, 32, dtype=torch.bool)

    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)

    assert output.shape == (4, 32, 1000)

    print("✓ BERT 模型测试通过")


def test_bfloat16():
    """测试 BFloat16 转换"""
    print("测试 BFloat16...")

    model = torch.nn.Linear(10, 5)
    model = convert_to_bfloat16(model)

    # 检查参数类型
    for p in model.parameters():
        assert p.dtype == torch.bfloat16

    print("✓ BFloat16 测试通过")


def test_fp8_support():
    """测试 FP8 支持检测"""
    print("测试 FP8 支持...")

    supported = check_fp8_support()
    print(f"  FP8 支持: {'是' if supported else '否'}")

    print("✓ FP8 支持检测通过")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print("运行基础功能测试")
    print("="*70 + "\n")

    tests = [
        test_config,
        test_polar_express,
        test_muon_optimizer,
        test_bert_model,
        test_bfloat16,
        test_fp8_support,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} 失败: {e}")
            failed += 1

    print("\n" + "="*70)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("="*70 + "\n")

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
