# -*- coding: utf-8 -*-
from typing import Optional, Tuple

import torch


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, seq_len):
    if len(x.shape) == 3:
        cos = cos[:, :seq_len, :]
        sin = sin[:, :seq_len, :]

        return (x * cos) + (rotate_half(x) * sin)
    elif len(x.shape) == 4:
        cos = cos[:, None, :seq_len, :]
        sin = sin[:, None, :seq_len, :]

        return (x * cos) + (rotate_half(x) * sin)
    else:
        raise ValueError(
            "Input tensor must have 3 or 4 dimensions, but got {}".format(x.shape)
        )


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim: int, base=10000, *_, **__):
        super().__init__()
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cos_sin_tables(self, x, seq_dimension=1):
        seq_len = x.shape[seq_dimension]

        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len != self._seq_len_cached or self._cos_cached.device != x.device:
            self._seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dimension], device=x.device).type_as(
                self.inv_freq
            )
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self._cos_cached = emb.cos()[None, :, :]
            self._sin_cached = emb.sin()[None, :, :]

        return self._cos_cached, self._sin_cached, seq_len

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self._cos_cached, self._sin_cached, seq_len = self._update_cos_sin_tables(
            k, seq_dimension=-2
        )

        return (
            apply_rotary_pos_emb(q, self._cos_cached, self._sin_cached, seq_len),
            apply_rotary_pos_emb(k, self._cos_cached, self._sin_cached, seq_len),
        )


class RotaryEmbedding2(torch.nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # For BC we register cos and sin cached
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=torch.int64
        ).type_as(self.inv_freq)
        t = t / self.scaling_factor
        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "_cos_cached", emb.cos().to(torch.get_default_dtype()), persistent=False
        )
        self.register_buffer(
            "_sin_cached", emb.sin().to(torch.get_default_dtype()), persistent=False
        )

    @property
    def sin_cached(self):
        return self._sin_cached

    @property
    def cos_cached(self):
        return self._cos_cached

    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`, *optional*):
                Deprecated and unused.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    @torch.no_grad()
    def forward(self, q, k):
        # x: [bs, num_attention_heads, seq_len, head_size]
        t = (
            torch.arange(q.shape[2], device=q.device)
            .type_as(self.inv_freq)
            .unsqueeze(0)
        )
        inv_freq_expanded = (
            self.inv_freq[None, :, None].float().expand(t.shape[0], -1, 1)
        )
        t_expanded = t[:, None, :].float()
        device_type = q.device.type
        device_type = (
            device_type
            if isinstance(device_type, str) and device_type != "mps"
            else "cpu"
        )
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ t_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        return self.apply_rotary_pos_emb(q, k, cos, sin)


class SFMRotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=16384,
        base=500000,
        # base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=True)
        self.max_seq_len_cached = max_position_embeddings

    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`, *optional*):
                Deprecated and unused.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        # cos = cos.unsqueeze(unsqueeze_dim)
        # sin = sin.unsqueeze(unsqueeze_dim)
        sLen = q.shape[-2]
        q_embed = (q * cos[:, :sLen, :]) + (rotate_half(q) * sin[:, :sLen, :])
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def forward(self, q, k, v, position_ids=None, nhead=1):
        """
        Args:
            q: [bs*num_attention_heads, tgt_len, head_size]
            k: [bs*num_attention_heads, seq_len, head_size]
            v: [bs*num_attention_heads, seq_len, head_size]
            position_ids: [bs, seq_len]
        return:
            q: [bs*num_attention_heads, tgt_len, head_size]
            k: [bs*num_attention_heads, seq_len, head_size]
        """
        with torch.no_grad():
            if position_ids is None:
                position_ids = (
                    torch.arange(v.shape[-2], device=q.device)
                    .type_as(self.inv_freq)
                    .unsqueeze(0)
                    .repeat(v.shape[0], 1)
                )
            else:
                max_seq_len = position_ids.size()[-1]
                position_ids = (
                    position_ids.unsqueeze(1)
                    .repeat(1, nhead, 1)
                    .reshape(-1, max_seq_len)
                )

            # x: [bs, num_attention_heads, seq_len, head_size]
            inv_freq_expanded = (
                self.inv_freq[None, :, None]
                .float()
                .expand(position_ids.shape[0], -1, 1)
            )
            position_ids_expanded = position_ids[:, None, :].float()
            device_type = v.device.type
            device_type = (
                device_type
                if isinstance(device_type, str) and device_type != "mps"
                else "cpu"
            )
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (
                    inv_freq_expanded.float() @ position_ids_expanded.float()
                ).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos()
                sin = emb.sin()

            cos, sin = cos.to(dtype=v.dtype), sin.to(dtype=v.dtype)

        q, k = self.apply_rotary_pos_emb(q, k, cos, sin)
        return q, k


class SFM2DRotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=16384,
        base=500000,
        # base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim // 2  # for 2D rope, half the dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=True)
        self.max_seq_len_cached = max_position_embeddings

    def apply_rotary_pos_emb(
        self, q, k, x_cos, x_sin, y_cos, y_sin, position_ids=None, unsqueeze_dim=1
    ):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`, *optional*):
                Deprecated and unused.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        # cos = cos.unsqueeze(unsqueeze_dim)
        # sin = sin.unsqueeze(unsqueeze_dim)
        sLen = q.shape[-2]
        sDep = q.shape[-3]
        q_x = q[..., : self.dim]
        q_y = q[..., self.dim :]
        k_x = k[..., : self.dim]
        k_y = k[..., self.dim :]
        q_x_embed = (q_x * x_cos[:, :, :sLen, :]) + (
            rotate_half(q_x) * x_sin[:, :, :sLen, :]
        )
        q_y_embed = (q_y * y_cos[:, :sDep, :, :]) + (
            rotate_half(q_y) * y_sin[:, :sDep, :, :]
        )
        k_x_embed = (k_x * x_cos) + (rotate_half(k_x) * x_sin)
        k_y_embed = (k_y * y_cos) + (rotate_half(k_y) * y_sin)
        q_embed = torch.cat((q_x_embed, q_y_embed), dim=-1)
        k_embed = torch.cat((k_x_embed, k_y_embed), dim=-1)
        return q_embed, k_embed

    def forward(self, q, k, position_ids=None, nhead=1):
        """
        Args:
            q: [bs*num_attention_heads, tgt_dep, tgt_len, head_size]
            k: [bs*num_attention_heads, tgt_dep, seq_len, head_size]
            v: [bs*num_attention_heads, tgt_dep, seq_len, head_size]
            position_ids: [bs, seq_len]
        return:
            q: [bs*num_attention_heads, tgt_dep, tgt_len, head_size]
            k: [bs*num_attention_heads, tgt_dep, seq_len, head_size]
        """
        with torch.no_grad():
            Bn, D, L = q.shape[0], q.shape[1], q.shape[2]
            device = q.device
            if position_ids is None:
                x = (
                    torch.arange(L, device=device)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                )
                x = x.expand(Bn, D, L, 1)
                y = (
                    torch.arange(D, device=device)
                    .unsqueeze(0)
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )
                y = y.expand(Bn, D, L, 1)
                position_ids = torch.cat((x, y), dim=-1)
            else:
                position_ids = (
                    position_ids.unsqueeze(1)
                    .repeat(1, nhead, 1, 1, 1)
                    .reshape(Bn, D, L, -1)
                )

            x_ids = position_ids[:, 0, :, 0]  # Bn L

            y_ids = position_ids[:, :, 0, 1]  # Bn D

            inv_freq_expanded = (
                self.inv_freq[None, :, None]
                .float()
                .expand(position_ids.shape[0], -1, 1)
            )
            x_ids_expanded = x_ids[:, None, :].float()
            y_ids_expanded = y_ids[:, None, :].float()
            device_type = q.device.type
            device_type = (
                device_type
                if isinstance(device_type, str) and device_type != "mps"
                else "cpu"
            )
            with torch.autocast(device_type=device_type, enabled=False):
                xfreqs = (inv_freq_expanded.float() @ x_ids_expanded.float()).transpose(
                    1, 2
                )
                x_emb = (
                    torch.cat((xfreqs, xfreqs), dim=-1).unsqueeze(1).repeat(1, D, 1, 1)
                )
                x_cos = x_emb.cos()
                x_sin = x_emb.sin()
                yfreqs = (inv_freq_expanded.float() @ y_ids_expanded.float()).transpose(
                    1, 2
                )
                y_emb = (
                    torch.cat((yfreqs, yfreqs), dim=-1).unsqueeze(2).repeat(1, 1, L, 1)
                )
                y_cos = y_emb.cos()
                y_sin = y_emb.sin()

            x_cos, x_sin, y_cos, y_sin = (
                x_cos.to(dtype=q.dtype),
                x_sin.to(dtype=q.dtype),
                y_cos.to(dtype=q.dtype),
                y_sin.to(dtype=q.dtype),
            )

        q, k = self.apply_rotary_pos_emb(q, k, x_cos, x_sin, y_cos, y_sin)
        return q, k
