import math

import torch
import torch.nn.functional as F
import torch_npu
import diffusers


def _attention(self, query, key, value, attention_mask=None):
    if self.upcast_attention:
        query = query.float()
        key = key.float()

    if query.dtype in (torch.float16, torch.bfloat16):
        query = query.reshape(query.shape[0] // self.heads, self.heads, query.shape[1], query.shape[2])
        key = key.reshape(key.shape[0] // self.heads, self.heads, key.shape[1], key.shape[2])
        value = value.reshape(value.shape[0] // self.heads, self.heads, value.shape[1], value.shape[2])
        hidden_states = torch_npu.npu_fusion_attention(
            query, key, value, self.heads, input_layout="BNSD",
            pse=None,
            atten_mask=attention_mask,
            scale=1.0 / math.sqrt(query.shape[-1]),
            pre_tockens=65536,
            next_tockens=65536,
            keep_prob=1,
            sync=False,
            inner_precise=0,
        )[0]

        hidden_states = hidden_states.reshape(hidden_states.shape[0] * self.heads, hidden_states.shape[2],
                                              hidden_states.shape[3])
    else:
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

    # reshape hidden_states
    hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
    return hidden_states


def geglu_forward(self, hidden_states):
    hidden_states = self.proj(hidden_states)
    return torch_npu.npu_geglu(hidden_states, dim=-1, approximate=1)[0]


def replace_with_torch_npu_flash_attention():
    diffusers.models.attention.CrossAttention._attention = _attention


def replace_with_torch_npu_geglu():
    diffusers.models.attention.GEGLU.forward = geglu_forward
