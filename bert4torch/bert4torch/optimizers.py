import torch
from torch.optim import Optimizer
from .backend import piecewise_linear


def extend_with_weight_decay(optimizer_class, exclude=['bias', 'LayerNorm']):
    """添加权重衰减（排除bias和LayerNorm）"""
    class OptimizerWithWeightDecay(optimizer_class):
        def __init__(self, *args, weight_decay=0.01, **kwargs):
            super().__init__(*args, **kwargs)
            self.weight_decay = weight_decay
            self.exclude = exclude

        def step(self, closure=None):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue

                    # check if should apply weight decay
                    apply_decay = True
                    for name in self.exclude:
                        if name in group.get('name', ''):
                            apply_decay = False
                            break

                    if apply_decay and self.weight_decay > 0:
                        p.data.add_(p.data, alpha=-self.weight_decay * group['lr'])

            return super().step(closure)

    return OptimizerWithWeightDecay


def extend_with_piecewise_linear_lr(optimizer_class):
    """添加分段线性学习率"""
    class OptimizerWithPiecewiseLinearLR(optimizer_class):
        def __init__(self, *args, lr_schedule=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.lr_schedule = lr_schedule or {}
            self.steps = 0

        def step(self, closure=None):
            self.steps += 1

            if self.lr_schedule:
                lr_multiplier = piecewise_linear(self.steps, self.lr_schedule)
                for group in self.param_groups:
                    group['lr'] = group.get('initial_lr', group['lr']) * lr_multiplier

            return super().step(closure)

    return OptimizerWithPiecewiseLinearLR


def extend_with_gradient_accumulation(optimizer_class):
    """添加梯度累积"""
    class OptimizerWithGradientAccumulation(optimizer_class):
        def __init__(self, *args, grad_accumulation_steps=1, **kwargs):
            super().__init__(*args, **kwargs)
            self.grad_accumulation_steps = grad_accumulation_steps
            self.accumulation_counter = 0

        def step(self, closure=None):
            self.accumulation_counter += 1

            if self.accumulation_counter % self.grad_accumulation_steps == 0:
                # scale gradients
                for group in self.param_groups:
                    for p in group['params']:
                        if p.grad is not None:
                            p.grad.data.div_(self.grad_accumulation_steps)

                result = super().step(closure)
                self.accumulation_counter = 0
                return result

            return None

    return OptimizerWithGradientAccumulation


def extend_with_exponential_moving_average(optimizer_class):
    """添加指数移动平均（EMA）"""
    class OptimizerWithEMA(optimizer_class):
        def __init__(self, *args, ema_decay=0.999, **kwargs):
            super().__init__(*args, **kwargs)
            self.ema_decay = ema_decay
            self.ema_params = {}

            # initialize EMA parameters
            for group in self.param_groups:
                for p in group['params']:
                    self.ema_params[id(p)] = p.data.clone()

        def step(self, closure=None):
            result = super().step(closure)

            # update EMA
            for group in self.param_groups:
                for p in group['params']:
                    if id(p) in self.ema_params:
                        self.ema_params[id(p)].mul_(self.ema_decay).add_(
                            p.data, alpha=1 - self.ema_decay
                        )

            return result

        def apply_ema(self):
            """应用EMA参数"""
            for group in self.param_groups:
                for p in group['params']:
                    if id(p) in self.ema_params:
                        p.data.copy_(self.ema_params[id(p)])

        def restore_params(self):
            """恢复原始参数"""
            for group in self.param_groups:
                for p in group['params']:
                    if id(p) in self.ema_params:
                        self.ema_params[id(p)] = p.data.clone()

    return OptimizerWithEMA


def extend_with_lookahead(optimizer_class):
    """添加Lookahead"""
    class OptimizerWithLookahead(optimizer_class):
        def __init__(self, *args, lookahead_k=5, lookahead_alpha=0.5, **kwargs):
            super().__init__(*args, **kwargs)
            self.lookahead_k = lookahead_k
            self.lookahead_alpha = lookahead_alpha
            self.lookahead_step = 0
            self.slow_params = {}

            # initialize slow parameters
            for group in self.param_groups:
                for p in group['params']:
                    self.slow_params[id(p)] = p.data.clone()

        def step(self, closure=None):
            result = super().step(closure)
            self.lookahead_step += 1

            if self.lookahead_step % self.lookahead_k == 0:
                # update slow parameters
                for group in self.param_groups:
                    for p in group['params']:
                        if id(p) in self.slow_params:
                            self.slow_params[id(p)].add_(
                                p.data - self.slow_params[id(p)],
                                alpha=self.lookahead_alpha
                            )
                            p.data.copy_(self.slow_params[id(p)])

            return result

    return OptimizerWithLookahead


class AdamW(Optimizer):
    """AdamW优化器"""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # state initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # decay the first and second moment
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = group['lr'] / bias_correction1
                denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])

                # update parameters
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # weight decay
                if group['weight_decay'] > 0:
                    p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])

        return loss


# 组合优化器示例
AdamWWithLR = extend_with_piecewise_linear_lr(AdamW)
AdamWWithEMA = extend_with_exponential_moving_average(AdamW)
AdamWWithGradAcc = extend_with_gradient_accumulation(AdamW)
AdamWWithLookahead = extend_with_lookahead(AdamW)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """带warmup的线性学习率调度"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step) /
            float(max(1, num_training_steps - num_warmup_steps))
        )

    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """带warmup的余弦学习率调度"""
    import math

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    from torch.optim.lr_scheduler import LambdaLR
    return LambdaLR(optimizer, lr_lambda)
