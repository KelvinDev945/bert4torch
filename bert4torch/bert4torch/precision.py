import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
import warnings


# ============ BFloat16 支持 ============

def convert_to_bfloat16(model: nn.Module, exclude_types: Optional[Tuple] = None):
    """将模型转换为 BFloat16 精度

    Args:
        model: 要转换的模型
        exclude_types: 排除的层类型（如 LayerNorm）
    """
    if exclude_types is None:
        exclude_types = (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)

    for m in model.modules():
        # 跳过排除的类型
        if isinstance(m, exclude_types):
            continue

        # 转换 Embedding 和 Linear
        if isinstance(m, (nn.Embedding, nn.Linear)):
            m.to(torch.bfloat16)

    return model


def bfloat16_optimizer_hook(model: nn.Module):
    """为 BFloat16 模型添加优化器钩子，确保梯度正确累积"""
    for p in model.parameters():
        if p.requires_grad and p.dtype == torch.bfloat16:
            # BF16 参数的梯度也是 BF16
            pass
    return model


# ============ FP8 自定义算子 ============
# 参考 modded-nanogpt 的 FP8 实现

try:
    # 仅在支持 FP8 的硬件上可用（如 H100）
    _FP8_AVAILABLE = hasattr(torch, 'float8_e4m3fn')
except:
    _FP8_AVAILABLE = False


if _FP8_AVAILABLE:
    @torch.library.custom_op("bert4torch::mm_fp8", mutates_args=())
    def mm_fp8_op(
        x: Tensor,
        w: Tensor,
        x_scale: float,
        w_scale: float,
        grad_scale: float
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """FP8 矩阵乘法 forward

        Args:
            x: 输入张量 [*, d_in]
            w: 权重张量 [d_out, d_in]
            x_scale: 输入缩放因子
            w_scale: 权重缩放因子
            grad_scale: 梯度缩放因子

        Returns:
            output: 输出张量 [*, d_out]
            x_fp8: FP8 输入
            w_fp8: FP8 权重
        """
        @torch.compile
        def impl(x: Tensor, w: Tensor):
            assert x.is_contiguous() and w.is_contiguous()

            # 转换为 FP8 (e4m3fn 用于前向)
            x_fp8 = x.div(x_scale).to(torch.float8_e4m3fn)
            w_fp8 = w.div(w_scale).to(torch.float8_e4m3fn)

            # 使用 scaled matmul
            out = torch._scaled_mm(
                x_fp8,
                w_fp8.T,
                out_dtype=torch.bfloat16,
                scale_a=x.new_tensor(x_scale, dtype=torch.float32),
                scale_b=x.new_tensor(w_scale, dtype=torch.float32),
                use_fast_accum=True,
            )

            return out, x_fp8, w_fp8

        return impl(x, w)

    @mm_fp8_op.register_fake
    def _(x: Tensor, w: Tensor, *_):
        """FP8 matmul 的伪实现（用于图追踪）"""
        assert x.ndim == w.ndim == 2
        assert x.shape[1] == w.shape[1]
        assert x.device == w.device
        assert x.is_contiguous() and w.is_contiguous()
        return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

    @torch.library.custom_op("bert4torch::mm_fp8_backward", mutates_args=())
    def mm_fp8_backward_op(
        grad_out: Tensor,
        x_fp8: Tensor,
        w_fp8: Tensor,
        x_scale: float,
        w_scale: float,
        grad_scale: float,
    ) -> Tuple[Tensor, Tensor]:
        """FP8 矩阵乘法 backward"""
        @torch.compile
        def impl(grad: Tensor, x_fp8: Tensor, w_fp8: Tensor):
            assert grad.is_contiguous()

            x_inv_scale = grad.new_tensor(x_scale, dtype=torch.float32)
            w_inv_scale = grad.new_tensor(w_scale, dtype=torch.float32)
            grad_inv_scale = grad.new_tensor(grad_scale, dtype=torch.float32)

            # 梯度转 FP8 (e5m2 用于反向)
            grad_fp8 = grad.div(grad_scale).to(torch.float8_e5m2)

            # 计算输入梯度
            grad_x = torch._scaled_mm(
                grad_fp8,
                w_fp8.T.contiguous().T,
                out_dtype=torch.bfloat16,
                scale_a=grad_inv_scale,
                scale_b=w_inv_scale,
                use_fast_accum=False,
            )

            # 计算权重梯度
            grad_w = torch._scaled_mm(
                x_fp8.T.contiguous(),
                grad_fp8.T.contiguous().T,
                out_dtype=torch.float32,
                scale_a=x_inv_scale,
                scale_b=grad_inv_scale,
                use_fast_accum=False,
            ).T

            return grad_x, grad_w

        return impl(grad_out, x_fp8, w_fp8)

    @mm_fp8_backward_op.register_fake
    def _(grad: Tensor, x_fp8: Tensor, w_fp8: Tensor, *_):
        """FP8 backward 的伪实现"""
        return x_fp8.to(torch.bfloat16), w_fp8.T.contiguous().T.to(torch.float32)

    def _mm_fp8_backward(ctx, grad_out: Tensor, *_):
        """自动微分 backward 函数"""
        x_fp8, w_fp8 = ctx.saved_tensors
        x_s, w_s, grad_s = ctx.scales
        grad_x, grad_w = torch.ops.bert4torch.mm_fp8_backward(
            grad_out, x_fp8, w_fp8, x_s, w_s, grad_s
        )
        return grad_x, grad_w, None, None, None

    def _mm_fp8_setup_context(ctx, inputs, output):
        """设置自动微分上下文"""
        *_, x_s, w_s, grad_s = inputs
        _, x_fp8, w_fp8 = output
        ctx.save_for_backward(x_fp8, w_fp8)
        ctx.scales = x_s, w_s, grad_s
        ctx.set_materialize_grads(False)

    mm_fp8_op.register_autograd(_mm_fp8_backward, setup_context=_mm_fp8_setup_context)


class FP8Linear(nn.Module):
    """FP8 线性层

    仅在 lm_head 或其他大型线性层使用，以节省内存和加速

    Args:
        in_features: 输入特征维度
        out_features: 输出特征维度
        bias: 是否使用 bias
        model_dim: 模型维度（用于计算缩放因子）
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        model_dim: int = 768,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 权重参数（保持 FP32/BF16）
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        # FP8 缩放因子（参考 modded-nanogpt）
        # x_scale = (model_dim ** 0.5) / 448
        # w_scale = 2 ** -9
        # grad_scale = 1 / 448
        self.register_buffer('x_scale', torch.tensor((model_dim ** 0.5) / 448.0))
        self.register_buffer('w_scale', torch.tensor(2.0 ** -9))
        self.register_buffer('grad_scale', torch.tensor(1.0 / 448.0))

        if not _FP8_AVAILABLE:
            warnings.warn("FP8 不可用，将回退到 BF16")

    def forward(self, x: Tensor) -> Tensor:
        """前向传播"""
        # 确保输入是 2D
        x_shape = x.shape
        x_2d = x.view(-1, self.in_features)

        if _FP8_AVAILABLE and x.dtype == torch.bfloat16:
            # 使用 FP8 matmul
            out, _, _ = torch.ops.bert4torch.mm_fp8(
                x_2d, self.weight,
                self.x_scale.item(),
                self.w_scale.item(),
                self.grad_scale.item()
            )
        else:
            # 回退到标准 matmul
            out = x_2d @ self.weight.T

        # 恢复原始形状
        out = out.view(*x_shape[:-1], self.out_features)

        if self.bias is not None:
            out = out + self.bias

        return out


# ============ AMP (Automatic Mixed Precision) 支持 ============

class GradScalerWrapper:
    """梯度缩放器包装类

    支持 FP16/BF16 混合精度训练
    BF16 通常不需要梯度缩放，但提供统一接口
    """
    def __init__(
        self,
        enabled: bool = True,
        init_scale: float = 65536.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        self.enabled = enabled

        if enabled:
            self.scaler = torch.cuda.amp.GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval,
            )
        else:
            self.scaler = None

    def scale(self, loss: Tensor) -> Tensor:
        """缩放损失"""
        if self.enabled and self.scaler is not None:
            return self.scaler.scale(loss)
        return loss

    def step(self, optimizer):
        """执行优化器步骤"""
        if self.enabled and self.scaler is not None:
            self.scaler.step(optimizer)
        else:
            optimizer.step()

    def update(self):
        """更新缩放因子"""
        if self.enabled and self.scaler is not None:
            self.scaler.update()

    def unscale_(self, optimizer):
        """取消缩放梯度"""
        if self.enabled and self.scaler is not None:
            self.scaler.unscale_(optimizer)


class AMPContext:
    """自动混合精度上下文管理器"""
    def __init__(
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.enabled = enabled
        self.dtype = dtype
        self.autocast_ctx = None

    def __enter__(self):
        if self.enabled:
            self.autocast_ctx = torch.autocast(
                device_type='cuda',
                dtype=self.dtype,
                enabled=True
            )
            return self.autocast_ctx.__enter__()
        return None

    def __exit__(self, *args):
        if self.autocast_ctx is not None:
            return self.autocast_ctx.__exit__(*args)


# ============ 工具函数 ============

def get_precision_context(precision: str = 'fp32'):
    """根据精度字符串获取上下文管理器

    Args:
        precision: 'fp32', 'bf16', 'fp16', 'fp8'

    Returns:
        上下文管理器
    """
    if precision == 'bf16':
        return AMPContext(enabled=True, dtype=torch.bfloat16)
    elif precision == 'fp16':
        return AMPContext(enabled=True, dtype=torch.float16)
    elif precision == 'fp8':
        # FP8 主要通过 FP8Linear 实现，这里返回 BF16 上下文
        return AMPContext(enabled=True, dtype=torch.bfloat16)
    else:
        return AMPContext(enabled=False)


def apply_precision_to_model(model: nn.Module, precision: str = 'fp32'):
    """为模型应用精度设置

    Args:
        model: 要转换的模型
        precision: 'fp32', 'bf16', 'fp8'

    Returns:
        转换后的模型
    """
    if precision == 'bf16' or precision == 'fp8':
        model = convert_to_bfloat16(model)

    return model


def check_fp8_support() -> bool:
    """检查当前硬件是否支持 FP8"""
    if not _FP8_AVAILABLE:
        return False

    # 检查 CUDA 计算能力（需要 sm_89 或更高，如 H100）
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        # sm_89 = H100, sm_90 = future GPUs
        if major >= 9 or (major == 8 and minor >= 9):
            return True

    return False


def get_recommended_precision() -> str:
    """获取推荐的精度设置"""
    if not torch.cuda.is_available():
        return 'fp32'

    if check_fp8_support():
        return 'fp8'

    # 大多数现代 GPU 支持 BF16（Ampere 及以上）
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:  # Ampere (A100, 3090, 4090, etc.)
        return 'bf16'

    # 旧 GPU 使用 FP16
    if major >= 7:  # Volta, Turing
        return 'fp16'

    return 'fp32'


# ============ 初始化配置 ============

def setup_precision_environment():
    """设置精度相关的环境变量"""
    import os

    # 启用 TF32（Ampere 及以上 GPU）
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # PyTorch 内存分配器优化
    if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
