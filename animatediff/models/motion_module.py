from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import torchvision

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention_processor import Attention
from diffusers.models.attention import FeedForward

from animatediff.utils.util import zero_rank_print

from einops import rearrange, repeat
import math, pdb
import random


def zero_module(module):
	# Zero out the parameters of a module and return it.
	for p in module.parameters():
		p.detach().zero_()
	return module


@dataclass
class TemporalTransformer3DModelOutput(BaseOutput):
	sample: torch.FloatTensor


def get_motion_module(
	in_channels,
	motion_module_type: str, 
	motion_module_kwargs: dict
):
	if motion_module_type == "Vanilla":
		return VanillaTemporalModule(in_channels=in_channels, **motion_module_kwargs)
	elif motion_module_type == "Conv":
		return ConvTemporalModule(in_channels=in_channels, **motion_module_kwargs)
	else:
		raise ValueError

class VanillaTemporalModule(nn.Module):
	def __init__(
		self,
		in_channels,
		num_attention_heads				   = 8,
		num_transformer_block			   = 2,
		attention_block_types			   =( "Temporal_Self", ),
		spatial_position_encoding		   = False,
		temporal_position_encoding		   = True,
		temporal_position_encoding_max_len = 32,
		temporal_attention_dim_div		   = 1,
		zero_initialize					   = True,
		
		causal_temporal_attention			= False,
		causal_temporal_attention_mask_type = "",
	):
		super().__init__()
		
		self.temporal_transformer = TemporalTransformer3DModel(
			in_channels=in_channels,
			num_attention_heads=num_attention_heads,
			attention_head_dim=in_channels // num_attention_heads // temporal_attention_dim_div,
			num_layers=num_transformer_block,
			attention_block_types=attention_block_types,
			temporal_position_encoding=temporal_position_encoding,
			temporal_position_encoding_max_len=temporal_position_encoding_max_len,
			spatial_position_encoding = spatial_position_encoding,
			causal_temporal_attention=causal_temporal_attention,
			causal_temporal_attention_mask_type=causal_temporal_attention_mask_type,
		)
		
		if zero_initialize:
			self.temporal_transformer.proj_out = zero_module(self.temporal_transformer.proj_out)

	def forward(self, input_tensor, temb=None, encoder_hidden_states=None, attention_mask=None):
		hidden_states = input_tensor
		hidden_states = self.temporal_transformer(hidden_states, encoder_hidden_states, attention_mask)

		output = hidden_states
		return output


class TemporalTransformer3DModel(nn.Module):	
	def __init__(
		self,
		in_channels,
		num_attention_heads,
		attention_head_dim,
		num_layers,
		attention_block_types			   = ( "Temporal_Self", "Temporal_Self", ),		   
		dropout							   = 0.0,
		norm_num_groups					   = 32,
		cross_attention_dim				   = 768,
		activation_fn					   = "geglu",
		attention_bias					   = False,
		upcast_attention				   = False,
		temporal_position_encoding		   = False,
		temporal_position_encoding_max_len = 32,
		spatial_position_encoding		   = False,
		
		causal_temporal_attention			= None,
		causal_temporal_attention_mask_type = "",
	):
		super().__init__()
		assert causal_temporal_attention is not None
		self.causal_temporal_attention			 = causal_temporal_attention

		assert (not causal_temporal_attention) or (causal_temporal_attention_mask_type != "")
		self.causal_temporal_attention_mask_type = causal_temporal_attention_mask_type
		self.causal_temporal_attention_mask		 = None
		self.spatial_position_encoding = spatial_position_encoding
		inner_dim = num_attention_heads * attention_head_dim

		self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
		self.proj_in = nn.Linear(in_channels, inner_dim)
		if spatial_position_encoding:
			self.pos_encoder_2d = PositionalEncoding2D(inner_dim)
		

		self.transformer_blocks = nn.ModuleList(
			[
				TemporalTransformerBlock(
					dim=inner_dim,
					num_attention_heads=num_attention_heads,
					attention_head_dim=attention_head_dim,
					attention_block_types=attention_block_types,
					dropout=dropout,
					norm_num_groups=norm_num_groups,
					cross_attention_dim=cross_attention_dim,
					activation_fn=activation_fn,
					attention_bias=attention_bias,
					upcast_attention=upcast_attention,
					temporal_position_encoding=temporal_position_encoding,
					temporal_position_encoding_max_len=temporal_position_encoding_max_len,
				)
				for d in range(num_layers)
			]
		)
		self.proj_out = nn.Linear(inner_dim, in_channels)
			
	def get_causal_temporal_attention_mask(self, hidden_states):
		batch_size, sequence_length, dim = hidden_states.shape
		
		if self.causal_temporal_attention_mask is None or self.causal_temporal_attention_mask.shape != (batch_size, sequence_length, sequence_length):
			zero_rank_print(f"build attn mask of type {self.causal_temporal_attention_mask_type}")
			if self.causal_temporal_attention_mask_type == "causal":
				# 1. vanilla causal mask
				mask = torch.tril(torch.ones(sequence_length, sequence_length))

			elif self.causal_temporal_attention_mask_type == "2-seq":
				# 2. 2-seq
				mask = torch.zeros(sequence_length, sequence_length)
				mask[:sequence_length // 2,  :sequence_length // 2]  = 1
				mask[-sequence_length // 2:, -sequence_length // 2:] = 1
			
			elif self.causal_temporal_attention_mask_type == "0-prev":
				# attn to the previous frame
				indices			= torch.arange(sequence_length)
				indices_prev	= indices - 1
				indices_prev[0] = 0
				mask = torch.zeros(sequence_length, sequence_length)
				mask[:,  0]					= 1.
				mask[indices, indices_prev] = 1.

			elif self.causal_temporal_attention_mask_type == "0":
				# only attn to first frame
				mask	  = torch.zeros(sequence_length, sequence_length)
				mask[:,0] = 1

			elif self.causal_temporal_attention_mask_type == "wo-self":
				indices = torch.arange(sequence_length)
				mask				   = torch.ones(sequence_length, sequence_length)
				mask[indices, indices] = 0

			elif self.causal_temporal_attention_mask_type == "circle":
				indices			= torch.arange(sequence_length)
				indices_prev	= indices - 1
				indices_prev[0] = 0

				mask = torch.eye(sequence_length)
				mask[indices, indices_prev] = 1
				mask[0,-1]					= 1

			else: raise ValueError

			# for sanity check
			if dim == 320: zero_rank_print(mask)

			# generate attention mask fron binary values
			mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
			mask = mask.unsqueeze(0)
			mask = mask.repeat(batch_size, 1, 1)

			self.causal_temporal_attention_mask = mask.to(hidden_states.device)
		
		return self.causal_temporal_attention_mask
	
	def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None):
		residual = hidden_states
		assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
		height, width = hidden_states.shape[-2:]
		
		hidden_states = self.norm(hidden_states)

		hidden_states = rearrange(hidden_states, "b c f h w -> (b h w) f c")
		hidden_states = self.proj_in(hidden_states)
		if self.spatial_position_encoding:

			video_length = hidden_states.shape[1]
			hidden_states = rearrange(hidden_states, "(b h w) f c -> (b f) h w c", h=height, w=width)
			pos_encoding = self.pos_encoder_2d(hidden_states)
			pos_encoding = rearrange(pos_encoding, "(b f) h w c -> (b h w) f c", f = video_length)
			hidden_states = rearrange(hidden_states, "(b f) h w c -> (b h w) f c", f=video_length)

		attention_mask = self.get_causal_temporal_attention_mask(hidden_states) if self.causal_temporal_attention else attention_mask

		# Transformer Blocks
		for block in self.transformer_blocks:
			if not self.spatial_position_encoding :
				pos_encoding = None
			
			hidden_states = block(hidden_states, pos_encoding=pos_encoding, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask)

		hidden_states = self.proj_out(hidden_states)

		hidden_states = rearrange(hidden_states, "(b h w) f c -> b c f h w", h=height, w=width)

		output = hidden_states + residual
		# output = hidden_states

		return output

class TemporalTransformerBlock(nn.Module):
	def __init__(
		self,
		dim,
		num_attention_heads,
		attention_head_dim,
		attention_block_types			   = ( "Temporal_Self", "Temporal_Self", ),
		dropout							   = 0.0,
		norm_num_groups					   = 32,
		cross_attention_dim				   = 768,
		activation_fn					   = "geglu",
		attention_bias					   = False,
		upcast_attention				   = False,
		temporal_position_encoding		   = False,
		temporal_position_encoding_max_len = 32,
	):
		super().__init__()

		attention_blocks = []
		norms = []
		
		for block_name in attention_block_types:
			attention_blocks.append(
				TemporalSelfAttention(
					attention_mode=block_name.split("_")[0],
					cross_attention_dim=cross_attention_dim if block_name.endswith("_Cross") else None,
					
					query_dim=dim,
					heads=num_attention_heads,
					dim_head=attention_head_dim,
					dropout=dropout,
					bias=attention_bias,
					upcast_attention=upcast_attention,
		
					temporal_position_encoding=temporal_position_encoding,
					temporal_position_encoding_max_len=temporal_position_encoding_max_len,
				)
			)
			norms.append(nn.LayerNorm(dim))
			
		self.attention_blocks = nn.ModuleList(attention_blocks)
		self.norms = nn.ModuleList(norms)
		
		self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
		self.ff_norm = nn.LayerNorm(dim)


	def forward(self, hidden_states, pos_encoding=None, encoder_hidden_states=None, attention_mask=None):
		for attention_block, norm in zip(self.attention_blocks, self.norms):
			if pos_encoding is not None:
				hidden_states += pos_encoding
			norm_hidden_states = norm(hidden_states)
			hidden_states = attention_block(
				norm_hidden_states,
				encoder_hidden_states=encoder_hidden_states,
				attention_mask=attention_mask,
			) + hidden_states

		hidden_states = self.ff(self.ff_norm(hidden_states)) + hidden_states
		
		output = hidden_states
		return output


def get_emb(sin_inp):
	"""
	Gets a base embedding for one dimension with sin and cos intertwined
	"""
	emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
	return torch.flatten(emb, -2, -1)

class PositionalEncoding2D(nn.Module):
	def __init__(self, channels):
		"""
		:param channels: The last dimension of the tensor you want to apply pos emb to.
		"""
		super(PositionalEncoding2D, self).__init__()
		self.org_channels = channels
		channels = int(np.ceil(channels / 4) * 2)
		self.channels = channels
		inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
		self.register_buffer("inv_freq", inv_freq)
		self.register_buffer("cached_penc", None)

	def forward(self, tensor):
		"""
		:param tensor: A 4d tensor of size (batch_size, x, y, ch)
		:return: Positional Encoding Matrix of size (batch_size, x, y, ch)
		"""
		if len(tensor.shape) != 4:
			raise RuntimeError("The input tensor has to be 4d!")

		if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
			return self.cached_penc

		self.cached_penc = None
		batch_size, x, y, orig_ch = tensor.shape
		pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
		pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
		sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
		sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
		emb_x = get_emb(sin_inp_x).unsqueeze(1)
		emb_y = get_emb(sin_inp_y)
		emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
			tensor.type()
		)
		emb[:, :, : self.channels] = emb_x
		emb[:, :, self.channels : 2 * self.channels] = emb_y

		self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
		return self.cached_penc

class PositionalEncoding(nn.Module):
	def __init__(
		self, 
		d_model,
		dropout = 0., 
		max_len = 32,
	):
		super().__init__()
		self.dropout = nn.Dropout(p=dropout)
		position = torch.arange(max_len).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
		pe = torch.zeros(1, max_len, d_model)
		pe[0, :, 0::2] = torch.sin(position * div_term)
		pe[0, :, 1::2] = torch.cos(position * div_term)
		self.register_buffer('pe', pe)

	def forward(self, x):
		# if x.size(1) < 16:
		# 	start_idx = random.randint(0, 12)
		# else:
		# 	start_idx = 0
		
		x = x + self.pe[:, :x.size(1)]
		return self.dropout(x)


class TemporalSelfAttention(Attention):
	def __init__(
			self,
			attention_mode					   = None,
			temporal_position_encoding		   = False,
			temporal_position_encoding_max_len = 32,
			*args, **kwargs
		):
		super().__init__(*args, **kwargs)
		assert attention_mode == "Temporal"

		self.pos_encoder = PositionalEncoding(
			kwargs["query_dim"],
			max_len=temporal_position_encoding_max_len
		) if temporal_position_encoding else None

	def set_use_memory_efficient_attention_xformers(
		self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
	):
		# disable motion module efficient xformers to avoid bad results, don't know why
		# TODO: fix this bug
		pass

	def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **cross_attention_kwargs):
		# The `Attention` class can call different attention processors / attention functions
		# here we simply pass along all tensors to the selected processor class
		# For standard processors that are defined here, `**cross_attention_kwargs` is empty

		# add position encoding
		hidden_states = self.pos_encoder(hidden_states)

		if hasattr(self.processor, "__call__"):
			return self.processor.__call__(
				self,
				hidden_states,
				encoder_hidden_states=None,
				attention_mask=attention_mask,
				**cross_attention_kwargs,
			)

		else:
			return self.processor(
				self,
				hidden_states,
				encoder_hidden_states=None,
				attention_mask=attention_mask,
				**cross_attention_kwargs,
			)
