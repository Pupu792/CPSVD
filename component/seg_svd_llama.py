import math
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.utils import logging
from transformers import LlamaConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    sin = torch.gather(sin.repeat(gather_indices.shape[0], 1, 1, 1), 2, gather_indices)
    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Seg_SVD_LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        rank: dict, 
        col_indices: dict, 
        svd_indices: dict
    ):
        super().__init__()
        
        for proj in ['up', 'gate', 'down']:
            setattr(self, f'col_indices_{proj}', col_indices[f'{proj}_proj'])
            setattr(self, f'svd_indices_{proj}', svd_indices[f'{proj}_proj'])
        
        self.is_pure_svd = {
            'up': len(self.col_indices_up) == 0,
            'gate': len(self.col_indices_gate) == 0,
            'down': len(self.col_indices_down) == 0,
        }
        rank_up, rank_gate, rank_down = rank['up_proj'], rank['gate_proj'], rank['down_proj']

        self.gate_u_proj = nn.Linear(rank_gate, intermediate_size, bias=False)
        self.gate_v_proj = nn.Linear(len(self.svd_indices_gate), rank_gate, bias=False)
        self.gate_col = nn.Linear(intermediate_size, len(self.col_indices_gate), bias=False)

        self.down_u_proj = nn.Linear(rank_down, hidden_size, bias=False)
        self.down_v_proj = nn.Linear(len(self.svd_indices_down), rank_down, bias=False)
        self.down_col = nn.Linear(hidden_size, len(self.col_indices_down), bias=False)

        self.up_u_proj = nn.Linear(rank_up, intermediate_size, bias=False)
        self.up_v_proj = nn.Linear(len(self.svd_indices_up), rank_up, bias=False)
        self.up_col = nn.Linear(intermediate_size, len(self.col_indices_gate), bias=False)
        
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        if self.is_pure_svd['up']:
            # 纯 SVD 模式
            up = self.up_u_proj(self.up_v_proj(x))
        else:
            # 混合模式
            x_svd_up = x[:, :, self.svd_indices_up]
            x_col_up = x[:, :, self.col_indices_up]
            svd_up = self.up_u_proj(self.up_v_proj(x_svd_up))
            col_up = self.up_col(x_col_up)
            up = svd_up + col_up
        
        if self.is_pure_svd['gate']:
            # 纯 SVD 模式
            gate = self.gate_u_proj(self.gate_v_proj(x))
        else:
            # 混合模式
            x_svd_gate = x[:, :, self.svd_indices_gate]
            x_col_gate = x[:, :, self.col_indices_gate]
            svd_gate = self.gate_u_proj(self.gate_v_proj(x_svd_gate))
            col_gate = self.gate_col(x_col_gate)
            gate = svd_gate + col_gate

        if self.is_pure_svd['down']:
            down =  self.down_u_proj(self.down_v_proj(self.act_fn(gate) * up))
        else:
            inter_x = self.act_fn(gate) * up
            inter_x_svd = inter_x[:, :, self.svd_indices_down]
            inter_x_col = inter_x[:, :, self.col_indices_down]
            svd_down = self.down_u_proj(self.down_v_proj(inter_x_svd))
            col_down = self.down_col(inter_x_col)
            down = svd_down + col_down
        
        return down


class Seg_SVD_LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, rank: dict, col_indices: dict, svd_indices: dict):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        for proj in ['q', 'k', 'v', 'o']:
            setattr(self, f'col_indices_{proj}', col_indices[f'{proj}_proj'])
            setattr(self, f'svd_indices_{proj}', svd_indices[f'{proj}_proj'])
        # self.ratio = ratio # 1 means no truncate, just keep normal attn

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        # low_rank = int(self.hidden_size * self.ratio/2)
        self.is_pure_svd = {
            'q': len(self.col_indices_q) == 0,
            'k': len(self.col_indices_k) == 0,
            'v': len(self.col_indices_v) == 0,
            'o': len(self.col_indices_o) == 0,
        }
        rank_q, rank_k, rank_v, rank_o = rank['q_proj'], rank['k_proj'], rank['v_proj'], rank['o_proj']

        self.q_u_proj = nn.Linear(rank_q, self.num_heads * self.head_dim, bias=False)
        self.q_v_proj = nn.Linear(len(self.svd_indices_q), rank_q, bias=False)
        self.q_col = nn.Linear(self.num_heads * self.head_dim, len(self.col_indices_q), bias=False)

        self.k_u_proj = nn.Linear(rank_k, self.num_heads * self.head_dim, bias=False)
        self.k_v_proj = nn.Linear(len(self.svd_indices_k), rank_k, bias=False)
        self.k_col = nn.Linear(self.num_heads * self.head_dim, len(self.col_indices_k), bias=False)

        self.v_u_proj = nn.Linear(rank_v, self.num_heads * self.head_dim, bias=False)
        self.v_v_proj = nn.Linear(len(self.svd_indices_v), rank_v, bias=False)
        self.v_col = nn.Linear(self.num_heads * self.head_dim, len(self.col_indices_v), bias=False)

        self.o_u_proj = nn.Linear(rank_o, self.hidden_size, bias=False)
        self.o_v_proj = nn.Linear(len(self.svd_indices_o), rank_o, bias=False)
        self.o_col = nn.Linear(self.hidden_size, len(self.col_indices_o), bias=False)

        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        if self.is_pure_svd['q']:
            # 纯 SVD 模式
            query_states = self.q_u_proj(self.q_v_proj(hidden_states))
        else:
            # 混合模式
            x_svd_q = hidden_states[:, :, self.svd_indices_q]
            x_col_q = hidden_states[:, :, self.col_indices_q]
            svd_q = self.q_u_proj(self.q_v_proj(x_svd_q))
            col_q = self.q_col(x_col_q)
            query_states = svd_q + col_q
        # 统一形状调整
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.is_pure_svd['k']:
            # 纯 SVD 模式
            key_states = self.k_u_proj(self.k_v_proj(hidden_states))
        else:
            # 混合模式
            x_svd_k = hidden_states[:, :, self.svd_indices_k]
            x_col_k = hidden_states[:, :, self.col_indices_k]
            svd_k = self.k_u_proj(self.k_v_proj(x_svd_k))
            col_k = self.k_col(x_col_k)
            key_states = svd_k + col_k
        # 统一形状调整
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.is_pure_svd['v']:
            # 纯 SVD 模式
            value_states = self.v_u_proj(self.v_v_proj(hidden_states))
        else:
            # 混合模式
            x_svd_v = hidden_states[:, :, self.svd_indices_v]
            x_col_v = hidden_states[:, :, self.col_indices_v]
            svd_v = self.v_u_proj(self.v_v_proj(x_svd_v))
            col_v = self.v_col(x_col_v)
            value_states = svd_v + col_v
        # 统一形状调整
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
 
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min, device=attn_weights.device))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, -1)

        if self.is_pure_svd['o']:
            attn_output = self.o_u_proj(self.o_v_proj(attn_output))
        else:
            x_svd_o = attn_output[:, :, self.svd_indices_o]
            x_col_o = attn_output[:, :, self.col_indices_o]
            svd_o = self.o_u_proj(self.o_v_proj(x_svd_o))
            col_o = self.o_col(x_col_o)
            attn_output = svd_o + col_o

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
    