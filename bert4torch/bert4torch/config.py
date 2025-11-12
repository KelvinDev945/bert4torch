import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Optional, Literal, Tuple, Dict, Any


@dataclass
class OptimizationConfig:
    """配置类，控制所有训练优化选项

    参考 modded-nanogpt 的优化策略，支持完整的训练加速配置
    """
    # ===== 精度设置 =====
    precision: Literal['fp32', 'bf16', 'fp8'] = 'fp32'
    use_amp: bool = False  # 自动混合精度
    fp8_lm_head: bool = False  # 仅对 lm_head 使用 FP8
    fp8_scale_x: float = 448.0 / (768 ** 0.5)  # FP8 x 缩放因子
    fp8_scale_w: float = 2 ** -9  # FP8 weight 缩放因子
    fp8_scale_grad: float = 1.0 / 448.0  # FP8 梯度缩放因子

    # ===== 编译优化 =====
    use_compile: bool = False
    compile_dynamic: bool = False  # 动态图模式
    compile_fullgraph: bool = True  # 完整图模式
    compile_mode: Literal['default', 'reduce-overhead', 'max-autotune'] = 'default'
    kernel_warmup_steps: int = 30  # 内核预热步数

    # ===== 优化器设置 =====
    optimizer_type: Literal['adam', 'adamw', 'muon', 'normuon', 'dist_adam'] = 'adamw'
    use_fused_adam: bool = True  # 使用融合 Adam

    # Adam/AdamW 参数
    adam_lr: float = 1e-4
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    adam_weight_decay: float = 0.01

    # Muon 参数
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_beta2: float = 0.95  # 低秩方差估计器
    muon_eps: float = 1e-8
    muon_rank: int = 128  # Adafactor 风格的低秩
    muon_momentum_warmup_steps: int = 300
    muon_momentum_cooldown_steps: int = 50
    use_polar_express: bool = True  # Polar Express 正交化
    use_newton_schulz: bool = False  # Newton-Schulz (旧版本)

    # 参数分组
    separate_optimizer_groups: bool = False  # 分离标量/矩阵参数

    # ===== 学习率调度 =====
    lr_schedule: Literal['constant', 'linear', 'cosine', 'piecewise'] = 'cosine'
    warmup_steps: int = 1000
    max_steps: int = 10000
    cooldown_frac: float = 0.5  # piecewise 冷却比例
    min_lr_ratio: float = 0.1  # 最小学习率比例

    # ===== 注意力机制 =====
    use_flash_attention: bool = False
    flash_attention_version: Literal['fa2', 'fa3'] = 'fa2'
    use_sliding_window: bool = False
    sliding_window_size: int = 2048
    window_size_warmup: bool = False
    attention_scale: float = 1.0  # 注意力缩放因子
    attention_scale_warmup: bool = False  # 从 0.1 开始预热

    # QK 归一化
    use_qk_norm: bool = False
    qk_norm_scale: float = 1.0

    # 注意力门控
    use_attention_gate: bool = False
    use_value_embeddings: bool = False

    # ===== 位置编码 =====
    use_yarn_rope: bool = False  # YaRN 动态 RoPE 缩放
    yarn_scale: float = 1.0
    yarn_temp: float = 1.0

    # ===== 架构增强 =====
    use_unet_skip: bool = False  # U-net 跳跃连接
    use_smear_module: bool = False  # Smear 模块
    use_backout_layers: bool = False  # Layer backout
    backout_num_layers: int = 8
    activation: Literal['gelu', 'relu', 'relu_squared', 'swiglu'] = 'gelu'

    # ===== 分布式训练 =====
    use_distributed: bool = False
    ddp_backend: Literal['nccl', 'gloo', 'mpi'] = 'nccl'
    async_grad_reduce: bool = False  # 异步梯度归约
    use_reduce_scatter: bool = False  # 使用 reduce_scatter 代替 all_reduce
    gradient_as_bucket_view: bool = True
    static_graph: bool = True

    # ===== 数据加载 =====
    use_async_dataloader: bool = False
    use_bos_alignment: bool = False  # BOS 对齐批次
    pin_memory: bool = True
    num_workers: int = 4
    prefetch_factor: int = 2

    # ===== 梯度设置 =====
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0  # 梯度裁剪

    # ===== 内存优化 =====
    use_gradient_checkpointing: bool = False
    use_cpu_offload: bool = False
    expandable_segments: bool = True  # PyTorch 内存分配器

    # ===== 批处理设置 =====
    batch_size: int = 32
    max_seq_len: int = 512
    alternating_optimizer_steps: bool = False  # Muon 每步更新，Adam 隔步更新

    # ===== 其他 =====
    seed: int = 42
    log_interval: int = 10
    eval_interval: int = 500
    save_interval: int = 1000
    device: str = 'cuda' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu'

    # ===== 实验记录 =====
    experiment_name: str = 'bert_fast_training'
    log_dir: str = './logs'
    save_dir: str = './checkpoints'

    def __post_init__(self):
        """验证配置参数"""
        if self.precision == 'fp8' and not self.use_compile:
            print("警告: FP8 通常需要 torch.compile 以获得最佳性能")

        if self.use_flash_attention and self.precision == 'fp32':
            print("警告: Flash Attention 建议使用 bf16 精度")

        if self.optimizer_type in ['muon', 'normuon'] and self.muon_lr > 0.1:
            print(f"警告: Muon 学习率 {self.muon_lr} 可能过大")

        if self.use_distributed and not self.use_compile:
            print("提示: 分布式训练配合 torch.compile 可获得更好性能")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'OptimizationConfig':
        """从 YAML 文件加载配置"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, yaml_path: str):
        """保存配置到 YAML 文件"""
        config_dict = asdict(self)
        # 将所有元组转换为列表，以避免YAML序列化问题
        def convert_tuples(obj):
            if isinstance(obj, dict):
                return {k: convert_tuples(v) for k, v in obj.items()}
            elif isinstance(obj, tuple):
                return list(obj)
            elif isinstance(obj, list):
                return [convert_tuples(item) for item in obj]
            else:
                return obj

        config_dict = convert_tuples(config_dict)
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    @classmethod
    def get_baseline_config(cls) -> 'OptimizationConfig':
        """获取基线配置（无优化）"""
        return cls(
            precision='fp32',
            use_compile=False,
            optimizer_type='adamw',
            use_flash_attention=False,
        )

    @classmethod
    def get_recommended_config(cls) -> 'OptimizationConfig':
        """获取推荐配置（主要优化）"""
        return cls(
            precision='bf16',
            use_compile=True,
            compile_fullgraph=True,
            optimizer_type='normuon',
            use_flash_attention=True,
            flash_attention_version='fa2',
            use_qk_norm=True,
            use_async_dataloader=True,
            muon_lr=0.02,
            adam_lr=0.008,
            lr_schedule='piecewise',
            cooldown_frac=0.5,
        )

    @classmethod
    def get_full_optimized_config(cls) -> 'OptimizationConfig':
        """获取完全优化配置（所有优化）"""
        return cls(
            precision='bf16',
            use_amp=True,
            fp8_lm_head=True,
            use_compile=True,
            compile_fullgraph=True,
            compile_mode='max-autotune',
            optimizer_type='normuon',
            use_polar_express=True,
            use_flash_attention=True,
            flash_attention_version='fa2',
            use_qk_norm=True,
            use_yarn_rope=True,
            use_async_dataloader=True,
            use_bos_alignment=True,
            async_grad_reduce=True,
            use_reduce_scatter=True,
            muon_lr=0.03,
            adam_lr=0.008,
            adam_betas=(0.65, 0.95),
            lr_schedule='piecewise',
            cooldown_frac=0.5,
            separate_optimizer_groups=True,
            alternating_optimizer_steps=True,
        )

    @classmethod
    def get_single_gpu_config(cls) -> 'OptimizationConfig':
        """获取单卡优化配置"""
        return cls(
            precision='bf16',
            use_compile=True,
            optimizer_type='normuon',
            use_flash_attention=True,
            use_qk_norm=True,
            muon_lr=0.02,
            lr_schedule='cosine',
            use_distributed=False,
        )

    @classmethod
    def get_multi_gpu_config(cls) -> 'OptimizationConfig':
        """获取多卡优化配置"""
        config = cls.get_full_optimized_config()
        config.use_distributed = True
        config.ddp_backend = 'nccl'
        config.async_grad_reduce = True
        config.use_reduce_scatter = True
        return config

    def get_summary(self) -> str:
        """获取配置摘要"""
        enabled_features = []

        if self.precision != 'fp32':
            enabled_features.append(f"{self.precision.upper()}")
        if self.use_compile:
            enabled_features.append("Compile")
        if self.optimizer_type in ['muon', 'normuon']:
            enabled_features.append(f"{self.optimizer_type.title()}")
        if self.use_flash_attention:
            enabled_features.append(f"FlashAttn-{self.flash_attention_version.upper()}")
        if self.use_qk_norm:
            enabled_features.append("QKNorm")
        if self.use_distributed:
            enabled_features.append("DDP")
        if self.fp8_lm_head:
            enabled_features.append("FP8-LMHead")
        if self.use_async_dataloader:
            enabled_features.append("AsyncData")

        if not enabled_features:
            return "Baseline (无优化)"

        return " + ".join(enabled_features)


def create_experiment_configs() -> Dict[str, OptimizationConfig]:
    """创建一组实验配置用于对比测试"""
    configs = {
        '01_baseline': OptimizationConfig.get_baseline_config(),
        '02_bf16': OptimizationConfig(precision='bf16'),
        '03_bf16_compile': OptimizationConfig(precision='bf16', use_compile=True),
        '04_bf16_muon': OptimizationConfig(precision='bf16', optimizer_type='normuon'),
        '05_bf16_flash': OptimizationConfig(precision='bf16', use_flash_attention=True),
        '06_bf16_compile_muon': OptimizationConfig(
            precision='bf16', use_compile=True, optimizer_type='normuon'
        ),
        '07_bf16_compile_flash': OptimizationConfig(
            precision='bf16', use_compile=True, use_flash_attention=True
        ),
        '08_bf16_muon_flash': OptimizationConfig(
            precision='bf16', optimizer_type='normuon', use_flash_attention=True
        ),
        '09_bf16_compile_muon_flash': OptimizationConfig(
            precision='bf16', use_compile=True, optimizer_type='normuon',
            use_flash_attention=True
        ),
        '10_bf16_compile_muon_flash_qknorm': OptimizationConfig(
            precision='bf16', use_compile=True, optimizer_type='normuon',
            use_flash_attention=True, use_qk_norm=True
        ),
        '11_recommended': OptimizationConfig.get_recommended_config(),
        '12_full_optimized': OptimizationConfig.get_full_optimized_config(),
        '13_bf16_compile_muon_asyncdata': OptimizationConfig(
            precision='bf16', use_compile=True, optimizer_type='normuon',
            use_async_dataloader=True
        ),
        '14_bf16_flash_qknorm_yarn': OptimizationConfig(
            precision='bf16', use_flash_attention=True, use_qk_norm=True,
            use_yarn_rope=True
        ),
        '15_bf16_compile_flash_fp8head': OptimizationConfig(
            precision='bf16', use_compile=True, use_flash_attention=True,
            fp8_lm_head=True
        ),
    }

    return configs
