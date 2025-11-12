__version__ = '0.1.0'

from .models import (
    BERT, RoFormer, GPT, T5,
    build_transformer_model
)

from .layers import (
    MultiHeadAttention, FeedForward, LayerNorm,
    Embedding, PositionEmbedding,
    GlobalPointer, CRF
)

from .tokenizers import Tokenizer, load_vocab

from .optimizers import (
    AdamW,
    Muon,
    NorMuon,
    polar_express,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)

from .snippets import (
    sequence_padding,
    truncate_sequences,
    DataGenerator,
    AutoRegressiveDecoder,
    ViterbiDecoder
)

from .config import (
    OptimizationConfig,
    create_experiment_configs
)

from .precision import (
    convert_to_bfloat16,
    FP8Linear,
    GradScalerWrapper,
    AMPContext,
    get_precision_context,
    apply_precision_to_model,
    check_fp8_support,
    get_recommended_precision,
    setup_precision_environment
)

from .distributed import (
    setup_distributed,
    cleanup_distributed,
    is_distributed,
    get_rank,
    get_world_size,
    wrap_model_ddp,
    DistributedLogger,
    DistributedContext
)

__all__ = [
    'BERT', 'RoFormer', 'GPT', 'T5',
    'build_transformer_model',
    'MultiHeadAttention', 'FeedForward', 'LayerNorm',
    'Embedding', 'PositionEmbedding',
    'GlobalPointer', 'CRF',
    'Tokenizer', 'load_vocab',
    'AdamW',
    'get_linear_schedule_with_warmup',
    'get_cosine_schedule_with_warmup',
    'sequence_padding',
    'truncate_sequences',
    'DataGenerator',
    'AutoRegressiveDecoder',
    'ViterbiDecoder',
    'OptimizationConfig',
    'create_experiment_configs',
    'convert_to_bfloat16',
    'FP8Linear',
    'GradScalerWrapper',
    'AMPContext',
    'get_precision_context',
    'apply_precision_to_model',
    'check_fp8_support',
    'get_recommended_precision',
    'setup_precision_environment',
    'setup_distributed',
    'cleanup_distributed',
    'is_distributed',
    'get_rank',
    'get_world_size',
    'wrap_model_ddp',
    'DistributedLogger',
    'DistributedContext',
]
