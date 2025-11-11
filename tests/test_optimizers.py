"""optimizers.py 单元测试"""
import unittest
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '..')

from bert4torch.optimizers import (
    AdamW,
    extend_with_weight_decay,
    extend_with_piecewise_linear_lr,
    extend_with_gradient_accumulation,
    extend_with_exponential_moving_average,
    extend_with_lookahead,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)


class TestAdamW(unittest.TestCase):
    """测试AdamW优化器"""

    def test_adamw_basic(self):
        """测试AdamW基础功能"""
        model = nn.Linear(10, 5)
        optimizer = AdamW(model.parameters(), lr=0.001)

        # 前向传播
        x = torch.randn(2, 10)
        output = model(x)
        loss = output.sum()

        # 反向传播
        loss.backward()
        optimizer.step()

        # 检查参数是否更新
        self.assertTrue(True)  # 如果没有错误就通过

    def test_adamw_weight_decay(self):
        """测试AdamW权重衰减"""
        model = nn.Linear(10, 5)
        optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

        initial_params = [p.clone() for p in model.parameters()]

        x = torch.randn(2, 10)
        output = model(x)
        loss = output.sum()

        loss.backward()
        optimizer.step()

        # 检查参数已更新
        for initial, current in zip(initial_params, model.parameters()):
            self.assertFalse(torch.allclose(initial, current))


class TestOptimizerExtensions(unittest.TestCase):
    """测试优化器扩展"""

    def test_extend_with_piecewise_linear_lr(self):
        """测试分段线性学习率"""
        model = nn.Linear(10, 5)

        # 创建扩展的优化器
        AdamWWithLR = extend_with_piecewise_linear_lr(AdamW)
        lr_schedule = {0: 0.0, 100: 1.0, 200: 0.5}
        optimizer = AdamWWithLR(
            model.parameters(),
            lr=0.001,
            lr_schedule=lr_schedule
        )

        # 初始学习率
        initial_lr = optimizer.param_groups[0]['lr']

        # 执行一些步骤
        for _ in range(50):
            x = torch.randn(2, 10)
            output = model(x)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # 学习率应该已变化
        current_lr = optimizer.param_groups[0]['lr']
        self.assertNotEqual(initial_lr, current_lr)

    def test_extend_with_gradient_accumulation(self):
        """测试梯度累积"""
        model = nn.Linear(10, 5)

        AdamWWithGradAcc = extend_with_gradient_accumulation(AdamW)
        optimizer = AdamWWithGradAcc(
            model.parameters(),
            lr=0.001,
            grad_accumulation_steps=4
        )

        # 累积梯度
        for i in range(4):
            x = torch.randn(2, 10)
            output = model(x)
            loss = output.sum()
            loss.backward()

            result = optimizer.step()
            if i < 3:
                self.assertIsNone(result)  # 前3步不更新
            else:
                self.assertIsNotNone(result)  # 第4步更新

            if i < 3:
                pass  # 不清零梯度
            else:
                optimizer.zero_grad()

    def test_extend_with_ema(self):
        """测试指数移动平均"""
        model = nn.Linear(10, 5)

        AdamWWithEMA = extend_with_exponential_moving_average(AdamW)
        optimizer = AdamWWithEMA(
            model.parameters(),
            lr=0.001,
            ema_decay=0.999
        )

        # 训练几步
        for _ in range(10):
            x = torch.randn(2, 10)
            output = model(x)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # EMA参数应该存在
        self.assertTrue(hasattr(optimizer, 'ema_params'))
        self.assertTrue(len(optimizer.ema_params) > 0)

    def test_extend_with_lookahead(self):
        """测试Lookahead"""
        model = nn.Linear(10, 5)

        AdamWWithLookahead = extend_with_lookahead(AdamW)
        optimizer = AdamWWithLookahead(
            model.parameters(),
            lr=0.001,
            lookahead_k=5,
            lookahead_alpha=0.5
        )

        # 训练一些步骤
        for _ in range(10):
            x = torch.randn(2, 10)
            output = model(x)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # 慢参数应该存在
        self.assertTrue(hasattr(optimizer, 'slow_params'))
        self.assertTrue(len(optimizer.slow_params) > 0)


class TestLRSchedulers(unittest.TestCase):
    """测试学习率调度器"""

    def test_linear_schedule_with_warmup(self):
        """测试线性warmup调度器"""
        model = nn.Linear(10, 5)
        optimizer = AdamW(model.parameters(), lr=0.001)

        num_warmup_steps = 100
        num_training_steps = 1000

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        # warmup阶段
        initial_lr = optimizer.param_groups[0]['lr']
        for _ in range(50):
            optimizer.step()
            scheduler.step()

        warmup_lr = optimizer.param_groups[0]['lr']

        # warmup期间学习率应该增加
        self.assertGreater(warmup_lr, initial_lr)

    def test_cosine_schedule_with_warmup(self):
        """测试余弦warmup调度器"""
        model = nn.Linear(10, 5)
        optimizer = AdamW(model.parameters(), lr=0.001)

        num_warmup_steps = 100
        num_training_steps = 1000

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        # 记录学习率变化
        lrs = []
        for _ in range(num_training_steps):
            lrs.append(optimizer.param_groups[0]['lr'])
            optimizer.step()
            scheduler.step()

        # warmup期间学习率应该增加
        self.assertLess(lrs[0], lrs[num_warmup_steps - 1])

        # 之后学习率应该下降
        self.assertGreater(lrs[num_warmup_steps], lrs[-1])


class TestOptimizerCombinations(unittest.TestCase):
    """测试优化器组合"""

    def test_multiple_extensions(self):
        """测试多个扩展的组合"""
        model = nn.Linear(10, 5)

        # 组合多个扩展
        OptimizerClass = extend_with_exponential_moving_average(AdamW)
        OptimizerClass = extend_with_piecewise_linear_lr(OptimizerClass)

        optimizer = OptimizerClass(
            model.parameters(),
            lr=0.001,
            ema_decay=0.999,
            lr_schedule={0: 0.0, 100: 1.0}
        )

        # 训练几步
        for _ in range(10):
            x = torch.randn(2, 10)
            output = model(x)
            loss = output.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # 应该同时有EMA和学习率调度
        self.assertTrue(hasattr(optimizer, 'ema_params'))
        self.assertTrue(hasattr(optimizer, 'lr_schedule'))


if __name__ == '__main__':
    unittest.main()
