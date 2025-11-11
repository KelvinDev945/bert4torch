import torch
import torch.nn as nn
import json
from .layers import *


class Transformer(nn.Module):
    """Transformer基类"""
    def __init__(self, vocab_size, hidden_size, num_hidden_layers, num_attention_heads,
                 intermediate_size, dropout=0.1, max_position=512, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.max_position = max_position

        # layers cache for reuse
        self.layers = {}

    def apply_embeddings(self, inputs):
        raise NotImplementedError

    def apply_main_layers(self, x, mask=None):
        raise NotImplementedError

    def apply_final_layers(self, x):
        return x

    def forward(self, *inputs):
        outputs = self.apply_embeddings(inputs)
        outputs = self.apply_main_layers(*outputs)
        outputs = self.apply_final_layers(outputs)
        return outputs

    def load_weights_from_checkpoint(self, checkpoint_path):
        """从checkpoint加载权重"""
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        mapping = self.variable_mapping()

        new_state_dict = {}
        for new_key, old_key in mapping.items():
            if old_key in state_dict:
                new_state_dict[new_key] = state_dict[old_key]

        self.load_state_dict(new_state_dict, strict=False)

    def variable_mapping(self):
        """构建变量映射"""
        return {}


class BERT(Transformer):
    def __init__(self, vocab_size, hidden_size=768, num_hidden_layers=12,
                 num_attention_heads=12, intermediate_size=3072, dropout=0.1,
                 max_position=512, segment_vocab_size=2, with_pool=False,
                 with_nsp=False, with_mlm=False, **kwargs):
        super().__init__(vocab_size, hidden_size, num_hidden_layers,
                        num_attention_heads, intermediate_size, dropout, max_position)

        self.segment_vocab_size = segment_vocab_size
        self.with_pool = with_pool
        self.with_nsp = with_nsp
        self.with_mlm = with_mlm

        # embeddings
        self.token_embedding = Embedding(vocab_size, hidden_size)
        self.position_embedding = PositionEmbedding(max_position, hidden_size)
        self.segment_embedding = Embedding(segment_vocab_size, hidden_size)
        self.emb_norm = LayerNorm(hidden_size)
        self.emb_dropout = nn.Dropout(dropout)

        # transformer layers
        self.encoders = nn.ModuleList([
            BERTLayer(hidden_size, num_attention_heads, intermediate_size, dropout)
            for _ in range(num_hidden_layers)
        ])

        # pooler
        if with_pool or with_nsp:
            self.pooler = nn.Linear(hidden_size, hidden_size)
            self.pooler_activation = nn.Tanh()

        # nsp
        if with_nsp:
            self.nsp = nn.Linear(hidden_size, 2)

        # mlm
        if with_mlm:
            self.mlm_dense = nn.Linear(hidden_size, hidden_size)
            self.mlm_norm = LayerNorm(hidden_size)
            self.mlm_decoder = nn.Linear(hidden_size, vocab_size, bias=False)
            self.mlm_decoder.weight = self.token_embedding.embedding.weight

    def apply_embeddings(self, inputs):
        token_ids = inputs[0]
        segment_ids = inputs[1] if len(inputs) > 1 else None

        x = self.token_embedding(token_ids)
        x = x + self.position_embedding(token_ids)
        if segment_ids is not None:
            x = x + self.segment_embedding(segment_ids)

        x = self.emb_norm(x)
        x = self.emb_dropout(x)

        # mask
        mask = self.compute_attention_mask(token_ids)
        return x, mask

    def apply_main_layers(self, x, mask=None):
        for layer in self.encoders:
            x = layer(x, mask)
        return x

    def apply_final_layers(self, x):
        if self.with_pool or self.with_nsp:
            pooled = self.pooler(x[:, 0])
            pooled = self.pooler_activation(pooled)

            if self.with_nsp:
                nsp_scores = self.nsp(pooled)
                return x, pooled, nsp_scores
            return x, pooled

        if self.with_mlm:
            mlm_hidden = self.mlm_dense(x)
            mlm_hidden = nn.GELU()(mlm_hidden)
            mlm_hidden = self.mlm_norm(mlm_hidden)
            mlm_scores = self.mlm_decoder(mlm_hidden)
            return mlm_scores

        return x

    def compute_attention_mask(self, token_ids):
        """计算attention mask"""
        mask = (token_ids != 0).long()
        mask = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
        mask = (1.0 - mask) * -10000.0
        return mask

    def variable_mapping(self):
        """BERT权重映射"""
        mapping = {
            'token_embedding.embedding.weight': 'bert.embeddings.word_embeddings.weight',
            'position_embedding.embedding.weight': 'bert.embeddings.position_embeddings.weight',
            'segment_embedding.embedding.weight': 'bert.embeddings.token_type_embeddings.weight',
            'emb_norm.weight': 'bert.embeddings.LayerNorm.weight',
            'emb_norm.bias': 'bert.embeddings.LayerNorm.bias',
        }

        for i in range(self.num_hidden_layers):
            prefix = f'encoders.{i}'
            old_prefix = f'bert.encoder.layer.{i}'
            mapping.update({
                f'{prefix}.attn.q.weight': f'{old_prefix}.attention.self.query.weight',
                f'{prefix}.attn.q.bias': f'{old_prefix}.attention.self.query.bias',
                f'{prefix}.attn.k.weight': f'{old_prefix}.attention.self.key.weight',
                f'{prefix}.attn.k.bias': f'{old_prefix}.attention.self.key.bias',
                f'{prefix}.attn.v.weight': f'{old_prefix}.attention.self.value.weight',
                f'{prefix}.attn.v.bias': f'{old_prefix}.attention.self.value.bias',
                f'{prefix}.attn.o.weight': f'{old_prefix}.attention.output.dense.weight',
                f'{prefix}.attn.o.bias': f'{old_prefix}.attention.output.dense.bias',
                f'{prefix}.norm1.weight': f'{old_prefix}.attention.output.LayerNorm.weight',
                f'{prefix}.norm1.bias': f'{old_prefix}.attention.output.LayerNorm.bias',
                f'{prefix}.ffn.dense1.weight': f'{old_prefix}.intermediate.dense.weight',
                f'{prefix}.ffn.dense1.bias': f'{old_prefix}.intermediate.dense.bias',
                f'{prefix}.ffn.dense2.weight': f'{old_prefix}.output.dense.weight',
                f'{prefix}.ffn.dense2.bias': f'{old_prefix}.output.dense.bias',
                f'{prefix}.norm2.weight': f'{old_prefix}.output.LayerNorm.weight',
                f'{prefix}.norm2.bias': f'{old_prefix}.output.LayerNorm.bias',
            })

        if self.with_pool or self.with_nsp:
            mapping.update({
                'pooler.weight': 'bert.pooler.dense.weight',
                'pooler.bias': 'bert.pooler.dense.bias',
            })

        return mapping


class BERTLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(hidden_size, num_attention_heads, dropout)
        self.norm1 = LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, intermediate_size, dropout)
        self.norm2 = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # self-attention
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


class RoFormer(BERT):
    """RoFormer with RoPE"""
    def __init__(self, *args, **kwargs):
        kwargs['use_rope'] = True
        super().__init__(*args, **kwargs)

        # replace position embedding
        del self.position_embedding
        self.rope = RoPEPositionEmbedding(self.hidden_size // self.num_attention_heads)

        # update encoder layers to use rope
        self.encoders = nn.ModuleList([
            RoFormerLayer(self.hidden_size, self.num_attention_heads,
                         self.intermediate_size, self.dropout, self.rope)
            for _ in range(self.num_hidden_layers)
        ])

    def apply_embeddings(self, inputs):
        token_ids = inputs[0]
        segment_ids = inputs[1] if len(inputs) > 1 else None

        x = self.token_embedding(token_ids)
        if segment_ids is not None:
            x = x + self.segment_embedding(segment_ids)

        x = self.emb_norm(x)
        x = self.emb_dropout(x)

        mask = self.compute_attention_mask(token_ids)
        return x, mask


class RoFormerLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout, rope):
        super().__init__()
        self.num_heads = num_attention_heads
        self.head_size = hidden_size // num_attention_heads
        self.rope = rope

        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.o = nn.Linear(hidden_size, hidden_size)

        self.norm1 = LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, intermediate_size, dropout)
        self.norm2 = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, hidden_size = x.shape

        # qkv
        q = self.q(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        k = self.k(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        v = self.v(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)

        # apply rope
        q, k = self.rope(q, k)

        # attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_size ** 0.5)
        if mask is not None:
            scores = scores + mask
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        attn_out = self.o(context)

        x = self.norm1(x + self.dropout(attn_out))

        # ffn
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x


class GPT(Transformer):
    """GPT模型"""
    def __init__(self, vocab_size, hidden_size=768, num_hidden_layers=12,
                 num_attention_heads=12, intermediate_size=3072, dropout=0.1,
                 max_position=512, **kwargs):
        super().__init__(vocab_size, hidden_size, num_hidden_layers,
                        num_attention_heads, intermediate_size, dropout, max_position)

        # embeddings
        self.token_embedding = Embedding(vocab_size, hidden_size)
        self.position_embedding = PositionEmbedding(max_position, hidden_size)
        self.emb_dropout = nn.Dropout(dropout)

        # transformer layers
        self.decoders = nn.ModuleList([
            GPTLayer(hidden_size, num_attention_heads, intermediate_size, dropout)
            for _ in range(num_hidden_layers)
        ])

        self.final_norm = LayerNorm(hidden_size)

    def apply_embeddings(self, inputs):
        token_ids = inputs[0]
        x = self.token_embedding(token_ids)
        x = x + self.position_embedding(token_ids)
        x = self.emb_dropout(x)

        # causal mask
        mask = self.compute_causal_mask(token_ids)
        return x, mask

    def apply_main_layers(self, x, mask=None):
        for layer in self.decoders:
            x = layer(x, mask)
        return x

    def apply_final_layers(self, x):
        x = self.final_norm(x)
        return x

    def compute_causal_mask(self, token_ids):
        """计算因果mask"""
        batch_size, seq_len = token_ids.shape
        mask = torch.tril(torch.ones(seq_len, seq_len, device=token_ids.device))
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        mask = (1.0 - mask) * -10000.0
        return mask


class GPTLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, dropout):
        super().__init__()
        self.norm1 = LayerNorm(hidden_size)
        self.attn = MultiHeadAttention(hidden_size, num_attention_heads, dropout)
        self.norm2 = LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, intermediate_size, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # pre-norm
        attn_out = self.attn(self.norm1(x), mask)
        x = x + self.dropout(attn_out)

        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)

        return x


class T5(Transformer):
    """T5 Encoder-Decoder模型"""
    def __init__(self, vocab_size, hidden_size=768, num_hidden_layers=12,
                 num_attention_heads=12, intermediate_size=3072, dropout=0.1, **kwargs):
        super().__init__(vocab_size, hidden_size, num_hidden_layers,
                        num_attention_heads, intermediate_size, dropout)

        # shared embedding
        self.shared = Embedding(vocab_size, hidden_size)

        # encoder
        self.encoder = T5Stack(hidden_size, num_hidden_layers, num_attention_heads,
                               intermediate_size, dropout, is_decoder=False)

        # decoder
        self.decoder = T5Stack(hidden_size, num_hidden_layers, num_attention_heads,
                               intermediate_size, dropout, is_decoder=True)

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.shared.embedding.weight

    def forward(self, input_ids, decoder_input_ids):
        encoder_outputs = self.encode(input_ids)
        decoder_outputs = self.decode(decoder_input_ids, encoder_outputs)
        lm_logits = self.lm_head(decoder_outputs)
        return lm_logits

    def encode(self, input_ids):
        x = self.shared(input_ids)
        mask = self.compute_attention_mask(input_ids)
        x = self.encoder(x, mask)
        return x

    def decode(self, decoder_input_ids, encoder_outputs, encoder_mask=None):
        x = self.shared(decoder_input_ids)
        decoder_mask = self.compute_causal_mask(decoder_input_ids)
        x = self.decoder(x, decoder_mask, encoder_outputs, encoder_mask)
        return x

    def compute_attention_mask(self, token_ids):
        mask = (token_ids != 0).long()
        mask = mask.unsqueeze(1).unsqueeze(2)
        mask = (1.0 - mask) * -10000.0
        return mask

    def compute_causal_mask(self, token_ids):
        batch_size, seq_len = token_ids.shape
        mask = torch.tril(torch.ones(seq_len, seq_len, device=token_ids.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = (1.0 - mask) * -10000.0
        return mask


class T5Stack(nn.Module):
    def __init__(self, hidden_size, num_layers, num_heads, intermediate_size,
                 dropout, is_decoder=False):
        super().__init__()
        self.is_decoder = is_decoder

        # relative position bias
        self.relative_attention_bias = RelativePositionEmbedding(num_heads)

        self.layers = nn.ModuleList([
            T5Layer(hidden_size, num_heads, intermediate_size, dropout, is_decoder)
            for _ in range(num_layers)
        ])

        self.final_norm = LayerNorm(hidden_size, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, encoder_hidden_states=None, encoder_mask=None):
        # compute position bias once
        seq_len = x.shape[1]
        position_bias = self.relative_attention_bias(seq_len, seq_len)

        for layer in self.layers:
            x = layer(x, mask, position_bias, encoder_hidden_states, encoder_mask)

        x = self.final_norm(x)
        x = self.dropout(x)
        return x


class T5Layer(nn.Module):
    def __init__(self, hidden_size, num_heads, intermediate_size, dropout, is_decoder):
        super().__init__()
        self.is_decoder = is_decoder

        self.self_attn = MultiHeadAttention(hidden_size, num_heads, dropout, position_bias=True)
        self.norm1 = LayerNorm(hidden_size, eps=1e-6)

        if is_decoder:
            self.cross_attn = MultiHeadAttention(hidden_size, num_heads, dropout)
            self.norm2 = LayerNorm(hidden_size, eps=1e-6)

        self.ffn = FeedForward(hidden_size, intermediate_size, dropout)
        self.norm3 = LayerNorm(hidden_size, eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, position_bias=None, encoder_hidden_states=None, encoder_mask=None):
        # self attention
        normed = self.norm1(x)
        attn_out = self.self_attn(normed, mask, position_bias=position_bias)
        x = x + self.dropout(attn_out)

        # cross attention (decoder only)
        if self.is_decoder and encoder_hidden_states is not None:
            normed = self.norm2(x)
            attn_out = self.cross_attn(normed, encoder_mask, kv=encoder_hidden_states)
            x = x + self.dropout(attn_out)

        # ffn
        normed = self.norm3(x)
        ffn_out = self.ffn(normed)
        x = x + self.dropout(ffn_out)

        return x


def build_transformer_model(config_path=None, checkpoint_path=None, model='bert', **kwargs):
    """统一的模型构建接口"""
    # load config
    config = {}
    if config_path:
        with open(config_path) as f:
            config = json.load(f)

    config.update(kwargs)

    # build model
    if model == 'bert':
        model = BERT(**config)
    elif model == 'roformer':
        model = RoFormer(**config)
    elif model == 'gpt':
        model = GPT(**config)
    elif model == 'gpt2':
        model = GPT(**config)
    elif model == 't5':
        model = T5(**config)
    else:
        raise ValueError(f'Unknown model: {model}')

    # load checkpoint
    if checkpoint_path:
        model.load_weights_from_checkpoint(checkpoint_path)

    return model
