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


# ============ Muon 优化器 ============

def polar_express(x: torch.Tensor, max_iter: int = 10, eps: float = 1e-7) -> torch.Tensor:
    """Polar Express 正交化

    使用快速收敛的极分解方法进行正交化
    比 Newton-Schulz 更稳定快速

    Args:
        x: 输入矩阵 [..., m, n]
        max_iter: 最大迭代次数
        eps: 收敛阈值

    Returns:
        正交化后的矩阵
    """
    # 保存原始形状
    orig_shape = x.shape

    # Reshape 为 2D if needed
    if x.ndim > 2:
        x = x.reshape(-1, x.shape[-2], x.shape[-1])

    # 初始化
    u = x.clone()

    for _ in range(max_iter):
        # U^T @ U
        uu = u.transpose(-2, -1) @ u

        # 计算逆平方根: (U^T @ U)^{-1/2}
        # 使用特征分解
        try:
            eigvals, eigvecs = torch.linalg.eigh(uu)
            eigvals = eigvals.clamp(min=eps)
            inv_sqrt = eigvecs @ torch.diag_embed(eigvals.rsqrt()) @ eigvecs.transpose(-2, -1)

            u_new = u @ inv_sqrt

            # 检查收敛
            diff = (u_new - u).norm() / (u.norm() + eps)
            u = u_new

            if diff < eps:
                break
        except:
            # 如果特征分解失败，使用简单的归一化
            u = torch.nn.functional.normalize(u, p=2, dim=-1)
            break

    # 恢复形状
    if len(orig_shape) > 2:
        u = u.reshape(orig_shape)

    return u


class Muon(Optimizer):
    """Muon 优化器

    结合动量和正交化的优化器
    参考: modded-nanogpt 的 NorMuon 实现
    简化版本，适合单卡训练

    Args:
        params: 参数
        lr: 学习率
        momentum: 动量系数
        beta2: 二阶动量系数（低秩方差估计）
        eps: 数值稳定性参数
        weight_decay: 权重衰减
        use_polar_express: 是否使用 Polar Express 正交化
    """
    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.0,
        use_polar_express=True,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            use_polar_express=use_polar_express,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # 状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(grad)

                    # 二阶动量 buffer（低秩）
                    if grad.ndim >= 2:
                        # 沿较小维度保存
                        if grad.shape[-2] >= grad.shape[-1]:
                            state['second_momentum_buffer'] = torch.zeros(
                                *grad.shape[:-1], 1, device=grad.device, dtype=grad.dtype
                            )
                        else:
                            state['second_momentum_buffer'] = torch.zeros(
                                *grad.shape[:-2], 1, grad.shape[-1], device=grad.device, dtype=grad.dtype
                            )
                    else:
                        state['second_momentum_buffer'] = torch.zeros_like(grad)

                state['step'] += 1
                momentum_buffer = state['momentum_buffer']
                second_momentum_buffer = state['second_momentum_buffer']

                # 一阶动量更新
                momentum_buffer.lerp_(grad, 1 - group['momentum'])
                updated_grad = grad.lerp(momentum_buffer, group['momentum'])

                # 正交化（仅对 2D 矩阵）
                if group['use_polar_express'] and updated_grad.ndim >= 2:
                    ortho_grad = polar_express(updated_grad)
                else:
                    ortho_grad = updated_grad

                # 二阶动量更新（NorMuon 的低秩方差估计）
                if ortho_grad.ndim >= 2:
                    # 计算梯度的平方均值
                    if ortho_grad.shape[-2] >= ortho_grad.shape[-1]:
                        v_mean = ortho_grad.square().mean(dim=-1, keepdim=True)
                    else:
                        v_mean = ortho_grad.square().mean(dim=-2, keepdim=True)

                    second_momentum_buffer.lerp_(v_mean, 1 - group['beta2'])
                    step_size = second_momentum_buffer.clamp_min(group['eps']).rsqrt()
                else:
                    step_size = 1.0

                # 参数更新
                p.data.add_(ortho_grad * step_size, alpha=-group['lr'])

                # 权重衰减
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

        return loss


class NorMuon(Muon):
    """NorMuon 优化器

    Muon 的归一化版本，默认启用 Polar Express
    """
    def __init__(self, params, lr=0.02, **kwargs):
        kwargs.setdefault('use_polar_express', True)
        super().__init__(params, lr=lr, **kwargs)
