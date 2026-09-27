# --------------------------------------------------------
# CAT-Seg: Cost Aggregation for Open-vocabulary Semantic Segmentation
# Licensed under The MIT License [see LICENSE for details]
# Written by Seokju Cho and Heeseong Shin
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from timm.layers import PatchEmbed, Mlp, DropPath, to_2tuple, to_ntuple, trunc_normal_, _assert

# Modified Swin Transformer blocks for guidance implementetion
# https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
# --------------------------------------------------------
# Spatial refinement (Replace Swin with LRSA + DWConv)
# Keep the same wrapper name / interface for minimal changes
# --------------------------------------------------------

class IRB(nn.Module):
    """
    Same IRB as LRFormer:
      1x1 conv -> GELU -> (x + DWConv(x)) -> 1x1 conv
    Input/Output are token format (B, N, C), requires (H, W) to reshape.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, ksize=3,
                 act_layer=nn.GELU, extra_act=False, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0)
        self.act = act_layer()
        self.act1 = act_layer() if extra_act else nn.Identity()
        self.conv = nn.Conv2d(hidden_features, hidden_features, kernel_size=ksize,
                              padding=ksize // 2, stride=1, groups=hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.conv(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x.reshape(B, -1, H * W).permute(0, 2, 1).contiguous()


class GuidedPoolingAttention(nn.Module):
    """
    LRFormer PoolingAttention variant, adapted to CAT-Seg guidance injection:
      - Q/K: concat([x, guidance]) -> Linear -> heads
      - V  : x only -> Linear -> heads
    K/V are built from pyramid pooled features (pooled_sizes), each enhanced by a DWConv (d_convs).
    Optional q_pooled_size: pool Q as well, then upsample output back to (H,W).
    """
    def __init__(self, dim, guidance_dim, num_heads=2,
                 qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.,
                 pooled_sizes=(8, 6, 5, 3),
                 q_pooled_size=1,
                 q_conv=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.guidance_dim = int(guidance_dim)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Q/K take (dim + guidance_dim), V takes dim
        self.q = nn.Linear(dim + self.guidance_dim, dim, bias=qkv_bias) if self.guidance_dim > 0 else nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim + self.guidance_dim, dim, bias=qkv_bias) if self.guidance_dim > 0 else nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pooled_sizes = list(pooled_sizes)
        self.eps = 0.001

        # normalize pooled x tokens (and pooled g tokens separately if needed)
        self.norm_x = nn.LayerNorm(dim)
        self.norm_g = nn.LayerNorm(self.guidance_dim) if self.guidance_dim > 0 else None

        self.q_pooled_size = int(q_pooled_size)

        # Optional q_conv branch (same spirit as LRFormer)
        if q_conv and self.q_pooled_size > 1:
            self.q_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
            self.q_norm = nn.LayerNorm(dim)
        else:
            self.q_conv = None
            self.q_norm = None

    def _aspect_pool_size(self, base, H, W):
        # keep aspect ratio similar to LRFormer
        if W >= H:
            return (base, int(round(W * float(base) / H + self.eps)))
        else:
            return (int(round(H * float(base) / W + self.eps)), base)

    def forward(self, x, H, W, guidance=None, d_convs=None):
        """
        x: (B, N, C), N=H*W
        guidance: (B, N, Cg) or None
        d_convs: list/ModuleList of DWConv layers, one per pooled_size
        """
        B, N, C = x.shape
        H, W = int(H), int(W)

        # -------- Q branch (optional pooling) --------
        if self.q_pooled_size > 1:
            q_size = self._aspect_pool_size(self.q_pooled_size, H, W)

            x2d = x.transpose(1, 2).reshape(B, C, H, W)
            q2d = F.adaptive_avg_pool2d(x2d, q_size)
            _, _, Hq, Wq = q2d.shape

            if self.q_conv is not None:
                q2d = q2d + self.q_conv(q2d)
                q_tok = self.q_norm(q2d.flatten(2).transpose(1, 2))  # (B, Hq*Wq, C)
            else:
                q_tok = q2d.flatten(2).transpose(1, 2)

            if self.guidance_dim > 0:
                if guidance is None:
                    raise ValueError("guidance_dim > 0 but guidance is None")
                g2d = guidance.transpose(1, 2).reshape(B, self.guidance_dim, H, W)
                gq2d = F.adaptive_avg_pool2d(g2d, q_size)
                gq_tok = gq2d.flatten(2).transpose(1, 2)
                q_in = torch.cat([q_tok, gq_tok], dim=-1)
            else:
                q_in = q_tok

            q = self.q(q_in).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
            Lq = q.shape[2]
        else:
            if self.guidance_dim > 0:
                if guidance is None:
                    raise ValueError("guidance_dim > 0 but guidance is None")
                q_in = torch.cat([x, guidance], dim=-1)
            else:
                q_in = x
            q = self.q(q_in).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
            Lq = N

        # -------- Pyramid pooling for K/V --------
        if d_convs is None:
            # If not provided, fallback to identity (still works, but less faithful)
            d_convs = [nn.Identity() for _ in self.pooled_sizes]

        x2d = x.permute(0, 2, 1).reshape(B, C, H, W)
        if self.guidance_dim > 0:
            if guidance is None:
                raise ValueError("guidance_dim > 0 but guidance is None")
            g2d = guidance.permute(0, 2, 1).reshape(B, self.guidance_dim, H, W)
        else:
            g2d = None

        pooled_x_list = []
        pooled_g_list = []

        for base, conv in zip(self.pooled_sizes, d_convs):
            psize = self._aspect_pool_size(int(base), H, W)

            px = F.adaptive_avg_pool2d(x2d, psize)
            px = px + conv(px)  # LRFormer: pool + DWConv(pool)
            pooled_x_list.append(px.flatten(2))  # (B, C, S)

            if self.guidance_dim > 0:
                pg = F.adaptive_avg_pool2d(g2d, psize)
                pooled_g_list.append(pg.flatten(2))  # (B, Cg, S)

        pooled_x = torch.cat(pooled_x_list, dim=2)  # (B, C, S_total)
        pooled_x_tok = pooled_x.permute(0, 2, 1).contiguous()  # (B, S_total, C)
        pooled_x_tok = self.norm_x(pooled_x_tok)

        if self.guidance_dim > 0:
            pooled_g = torch.cat(pooled_g_list, dim=2)  # (B, Cg, S_total)
            pooled_g_tok = pooled_g.permute(0, 2, 1).contiguous()  # (B, S_total, Cg)
            pooled_g_tok = self.norm_g(pooled_g_tok)
            k_in = torch.cat([pooled_x_tok, pooled_g_tok], dim=-1)  # (B, S_total, C+Cg)
        else:
            k_in = pooled_x_tok

        k = self.k(k_in).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        v = self.v(pooled_x_tok).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        # -------- Attention --------
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, h, Lq, S_total)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v  # (B, h, Lq, dh)
        out = out.transpose(1, 2).reshape(B, Lq, C).contiguous()
        out = self.proj_drop(self.proj(out))

        # -------- Upsample for residual if q was pooled --------
        if self.q_pooled_size > 1:
            # out: (B, Hq*Wq, C) -> (B, C, Hq, Wq) -> upsample -> (B, H*W, C)
            out2d = out.transpose(1, 2).reshape(B, C, Hq, Wq)
            out2d = F.interpolate(out2d, size=(H, W), mode='bilinear', align_corners=True)
            out = out2d.flatten(2).transpose(1, 2).contiguous()

        # ensure output is (B, H*W, C)
        if out.shape[1] != H * W:
            raise RuntimeError(f"Unexpected out length: {out.shape[1]} vs H*W={H*W}")
        return out


class LRSARefineBlock(nn.Module):
    """
    LRFormer-style Block: (CPE) + PoolingAttention + IRB, with DropPath and optional LayerScale.
    This is used to replace SwinTransformerBlock in CAT-Seg spatial refinement.
    """
    def __init__(self, dim, guidance_dim, num_heads,
                 pooled_sizes=(8, 6, 5, 3),
                 q_pooled_size=1,
                 q_conv=False,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 drop=0.0,
                 attn_drop=0.0,
                 drop_path=0.0,
                 norm_layer=nn.LayerNorm,
                 ls=False,
                 extra_act=False,
                 use_prenorm=False):
        super().__init__()

        self.pre_norm = nn.BatchNorm2d(dim) if use_prenorm else nn.Identity()

        self.cpe = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)  # CPE (DWConv)

        self.norm1 = norm_layer(dim)
        self.attn = GuidedPoolingAttention(
            dim=dim,
            guidance_dim=guidance_dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=None,
            attn_drop=attn_drop,
            proj_drop=drop,
            pooled_sizes=pooled_sizes,
            q_pooled_size=q_pooled_size,
            q_conv=q_conv,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = IRB(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=nn.GELU, drop=drop, ksize=3, extra_act=extra_act)

        self.ls = ls
        if self.ls:
            layer_scale_init_value = 1e-6
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

        # per pooled size DWConvs (LRFormer: d_convsX passed in; here we internalize for CAT-Seg)
        self.d_convs = nn.ModuleList([
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim) for _ in pooled_sizes
        ])

    def forward(self, x, H, W, guidance=None):
        """
        x: (B, H*W, C)
        guidance: (B, H*W, Cg) or None
        """
        B, N, C = x.shape
        assert N == H * W, "token length mismatch"

        # CPE
        x2d = x.permute(0, 2, 1).reshape(B, C, H, W)
        x2d = self.pre_norm(x2d)
        x2d = self.cpe(x2d) + x2d
        x = x2d.flatten(2).transpose(1, 2).contiguous()

        # Attention + IRB
        if self.ls:
            x = x + self.drop_path(self.layer_scale_1[None, None, :] *
                                   self.attn(self.norm1(x), H, W, guidance=guidance, d_convs=self.d_convs))
            x = x + self.drop_path(self.layer_scale_2[None, None, :] *
                                   self.mlp(self.norm2(x), H, W))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x), H, W, guidance=guidance, d_convs=self.d_convs))
            x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


# class SwinTransformerBlockWrapper(nn.Module):
#     """
#     Keep the original name / interface used by CAT-Seg.
#     Replace internal Swin blocks with LRFormer-style LRSARefineBlock.

#     Mapping from CAT-Seg config:
#       WINDOW_SIZES (int)  -> here interpreted as q_pooled_size target (like LRFormer q_pooled_size).
#       pooled_sizes        -> derived from LRFormer base [11,8,6,4] scaled by q_pooled_size/16.

#     For your setting: feature_resolution=24, WINDOW_SIZES=12
#       q_pooled_size = 12
#       pooled_sizes = round([11,8,6,4] * 12/16) -> [8,6,5,3]
#     """
#     def __init__(self, dim, appearance_guidance_dim, input_resolution, nheads=4, window_size=12, pad_len=0):
#         super().__init__()
#         H, W = to_2tuple(input_resolution)

#         if isinstance(window_size, (tuple, list)):
#             window_size = window_size[0]
#         q_pooled_size = int(window_size)
#         q_pooled_size = max(1, min(q_pooled_size, max(H, W)))  # clamp

#         base = [11, 8, 6, 4]
#         pooled_sizes = [max(1, int(round(t * q_pooled_size / 16.0 + 0.001))) for t in base]
#         pooled_sizes = [min(s, max(H, W)) for s in pooled_sizes]

#         self.guidance_norm = nn.LayerNorm(appearance_guidance_dim) if appearance_guidance_dim > 0 else None

#         self.block_1 = LRSARefineBlock(
#             dim=dim, guidance_dim=appearance_guidance_dim,
#             num_heads=nheads,
#             pooled_sizes=pooled_sizes,
#             q_pooled_size=q_pooled_size,
#             q_conv=False,
#             mlp_ratio=4.0,
#             qkv_bias=True,
#             drop=0.0, attn_drop=0.0, drop_path=0.0,
#             norm_layer=nn.LayerNorm,
#             ls=False,           # you can set True if you want LRFormer-like layerscale
#             extra_act=False,
#             use_prenorm=False,
#         )
#         self.block_2 = LRSARefineBlock(
#             dim=dim, guidance_dim=appearance_guidance_dim,
#             num_heads=nheads,
#             pooled_sizes=pooled_sizes,
#             q_pooled_size=q_pooled_size,
#             q_conv=False,
#             mlp_ratio=4.0,
#             qkv_bias=True,
#             drop=0.0, attn_drop=0.0, drop_path=0.0,
#             norm_layer=nn.LayerNorm,
#             ls=False,
#             extra_act=False,
#             use_prenorm=False,
#         )

#         self.pad_len = pad_len
#         self.padding_tokens = nn.Parameter(torch.zeros(1, 1, dim)) if pad_len > 0 else None
#         self.padding_guidance = nn.Parameter(torch.zeros(1, 1, appearance_guidance_dim)) if pad_len > 0 and appearance_guidance_dim > 0 else None

#         self.input_resolution = (H, W)

#     def forward(self, x, appearance_guidance):
#         """
#         x: B C T H W
#         appearance_guidance: B Cg H W
#         """
#         B, C, T, H, W = x.shape
#         assert (H, W) == self.input_resolution, f"Expected {self.input_resolution}, got {(H, W)}"

#         x = rearrange(x, 'B C T H W -> (B T) (H W) C')

#         g = None
#         if appearance_guidance is not None:
#             g = repeat(appearance_guidance, 'B C H W -> (B T) (H W) C', T=T)
#             if self.guidance_norm is not None:
#                 g = self.guidance_norm(g)

#         x = self.block_1(x, H, W, guidance=g)
#         x = self.block_2(x, H, W, guidance=g)

#         x = rearrange(x, '(B T) (H W) C -> B C T H W', B=B, T=T, H=H, W=W)
#         return x

class SwinTransformerBlockWrapper(nn.Module):
    def __init__(
        self,
        dim,
        appearance_guidance_dim,
        input_resolution,
        nheads=4,
        window_size=12,
        pad_len=0
    ):
        super().__init__()
        H, W = to_2tuple(input_resolution)
        R = max(H, W)

        if isinstance(window_size, (tuple, list)):
            window_size = window_size[0]

        # ---- q1=q2 ----
        q1 = int(window_size)
        q1 = max(1, min(q1, R))
        q2 = q1

        # ---- pooled_sizes for block1 (保持你现在的推导) ----
        base = [11, 8, 6, 4]
        pooled1 = [max(1, int(round(t * q1 / 16.0 + 0.001))) for t in base]
        pooled1 = [min(s, R) for s in pooled1]   # e.g. [8,6,5,3] for q1=12,R=24

        # ---- pooled_sizes for block2: 更细一点（推荐）----
        # 方案A：直接用 [12,8,6,4]（对 24x24 很合适）
        pooled2 = [min(R, s) for s in [12, 8, 6, 4]]

        # 方案B（可选）：在 pooled1 后面加一个更细 level（非常稳）
        # pooled2 = pooled1 + [2]

        self.guidance_norm = nn.LayerNorm(appearance_guidance_dim) if appearance_guidance_dim > 0 else None

        self.block_1 = LRSARefineBlock(
            dim=dim,
            guidance_dim=appearance_guidance_dim,
            num_heads=nheads,
            pooled_sizes=pooled1,
            q_pooled_size=q1,
            q_conv=False,          # block1 保持稳健
            mlp_ratio=4.0,
            qkv_bias=True,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            ls=False,
            extra_act=False,
            use_prenorm=False,
        )

        self.block_2 = LRSARefineBlock(
            dim=dim,
            guidance_dim=appearance_guidance_dim,
            num_heads=nheads,
            pooled_sizes=pooled2,
            q_pooled_size=q2,      # q2=q1
            q_conv=True,           # 关键：打开
            mlp_ratio=4.0,
            qkv_bias=True,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            ls=True,               # 推荐打开，抑制 block2 过强导致的噪声
            extra_act=False,
            use_prenorm=False,
        )

        # shift size (仿 Swin)
        self.shift_size = int(window_size) // 2  # e.g. 6 for window_size=12

        self.pad_len = pad_len
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, dim)) if pad_len > 0 else None
        self.padding_guidance = (
            nn.Parameter(torch.zeros(1, 1, appearance_guidance_dim))
            if pad_len > 0 and appearance_guidance_dim > 0 else None
        )
        self.input_resolution = (H, W)

    def forward(self, x, appearance_guidance):
        """
        x: B C T H W
        appearance_guidance: B Cg H W
        """
        B, C, T, H, W = x.shape
        assert (H, W) == self.input_resolution, f"Expected {self.input_resolution}, got {(H, W)}"

        x = rearrange(x, 'B C T H W -> (B T) (H W) C')

        g = None
        if appearance_guidance is not None:
            g = repeat(appearance_guidance, 'B C H W -> (B T) (H W) C', T=T)
            if self.guidance_norm is not None:
                g = self.guidance_norm(g)

        # block1
        x = self.block_1(x, H, W, guidance=g)

        # ---- Swin-style shift for block2 ----
        if self.shift_size > 0:
            BT = x.shape[0]
            x2d = x.view(BT, H, W, C)
            x2d = torch.roll(x2d, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            x = x2d.view(BT, H * W, C)

            if g is not None:
                Cg = g.shape[-1]
                g2d = g.view(BT, H, W, Cg)
                g2d = torch.roll(g2d, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                g = g2d.view(BT, H * W, Cg)

        x = self.block_2(x, H, W, guidance=g)

        if self.shift_size > 0:
            BT = x.shape[0]
            x2d = x.view(BT, H, W, C)
            x2d = torch.roll(x2d, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            x = x2d.view(BT, H * W, C)

        x = rearrange(x, '(B T) (H W) C -> B C T H W', B=B, T=T, H=H, W=W)
        return x


# spatial attention end


def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class FullAttention(nn.Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim, guidance_dim, nheads=8, attention_type='linear'):
        super().__init__()
        self.nheads = nheads
        self.q = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim + guidance_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)

        if attention_type == 'linear':
            self.attention = LinearAttention()
        elif attention_type == 'full':
            self.attention = FullAttention()
        else:
            raise NotImplementedError
    
    def forward(self, x, guidance):
        """
        Arguments:
            x: B, L, C
            guidance: B, L, C
        """
        q = self.q(torch.cat([x, guidance], dim=-1)) if guidance is not None else self.q(x)
        k = self.k(torch.cat([x, guidance], dim=-1)) if guidance is not None else self.k(x)
        v = self.v(x)

        q = rearrange(q, 'B L (H D) -> B L H D', H=self.nheads)
        k = rearrange(k, 'B S (H D) -> B S H D', H=self.nheads)
        v = rearrange(v, 'B S (H D) -> B S H D', H=self.nheads)

        out = self.attention(q, k, v)
        out = rearrange(out, 'B L H D -> B L (H D)')
        return out


class ClassTransformerLayer(nn.Module):
    def __init__(self, hidden_dim=64, guidance_dim=64, nheads=8, attention_type='linear', pooling_size=(4, 4), pad_len=256) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(pooling_size) if pooling_size is not None else nn.Identity()
        self.attention = AttentionLayer(hidden_dim, guidance_dim, nheads=nheads, attention_type=attention_type)
        self.MLP = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.pad_len = pad_len
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, hidden_dim)) if pad_len > 0 else None
        self.padding_guidance = nn.Parameter(torch.zeros(1, 1, guidance_dim)) if pad_len > 0 and guidance_dim > 0 else None
    
    def pool_features(self, x):
        """
        Intermediate pooling layer for computational efficiency.
        Arguments:
            x: B, C, T, H, W
        """
        B = x.size(0)
        x = rearrange(x, 'B C T H W -> (B T) C H W')
        x = self.pool(x)
        x = rearrange(x, '(B T) C H W -> B C T H W', B=B)
        return x

    def forward(self, x, guidance):
        """
        Arguments:
            x: B, C, T, H, W
            guidance: B, T, C
        """
        B, C, T, H, W = x.size()
        x_pool = self.pool_features(x)
        *_, H_pool, W_pool = x_pool.size()
        
        if self.padding_tokens is not None:
            orig_len = x.size(2)
            if orig_len < self.pad_len:
                # pad to pad_len
                padding_tokens = repeat(self.padding_tokens, '1 1 C -> B C T H W', B=B, T=self.pad_len - orig_len, H=H_pool, W=W_pool)
                x_pool = torch.cat([x_pool, padding_tokens], dim=2)

        x_pool = rearrange(x_pool, 'B C T H W -> (B H W) T C')
        if guidance is not None:
            if self.padding_guidance is not None:
                if orig_len < self.pad_len:
                    padding_guidance = repeat(self.padding_guidance, '1 1 C -> B T C', B=B, T=self.pad_len - orig_len)
                    guidance = torch.cat([guidance, padding_guidance], dim=1)
            guidance = repeat(guidance, 'B T C -> (B H W) T C', H=H_pool, W=W_pool)

        x_pool = x_pool + self.attention(self.norm1(x_pool), guidance) # Attention
        x_pool = x_pool + self.MLP(self.norm2(x_pool)) # MLP

        x_pool = rearrange(x_pool, '(B H W) T C -> (B T) C H W', H=H_pool, W=W_pool)
        x_pool = F.interpolate(x_pool, size=(H, W), mode='bilinear', align_corners=True)
        x_pool = rearrange(x_pool, '(B T) C H W -> B C T H W', B=B)

        if self.padding_tokens is not None:
            if orig_len < self.pad_len:
                x_pool = x_pool[:, :, :orig_len]

        x = x + x_pool # Residual
        return x


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AggregatorLayer(nn.Module):
    def __init__(self, hidden_dim=64, text_guidance_dim=512, appearance_guidance=512, nheads=4, input_resolution=(20, 20), pooling_size=(5, 5), window_size=(10, 10), attention_type='linear', pad_len=256) -> None:
        super().__init__()
        self.swin_block = SwinTransformerBlockWrapper(hidden_dim, appearance_guidance, input_resolution, nheads, window_size)
        self.attention = ClassTransformerLayer(hidden_dim, text_guidance_dim, nheads=nheads, attention_type=attention_type, pooling_size=pooling_size, pad_len=pad_len)

    def forward(self, x, appearance_guidance, text_guidance):
        """
        Arguments:
            x: B C T H W
        """
        x = self.swin_block(x, appearance_guidance)
        x = self.attention(x, text_guidance)
        return x


class AggregatorResNetLayer(nn.Module):
    def __init__(self, hidden_dim=64, appearance_guidance=512) -> None:
        super().__init__()
        self.conv_linear = nn.Conv2d(hidden_dim + appearance_guidance, hidden_dim, kernel_size=1, stride=1)
        self.conv_layer = Bottleneck(hidden_dim, hidden_dim // 4)


    def forward(self, x, appearance_guidance):
        """
        Arguments:
            x: B C T H W
        """
        B, T = x.size(0), x.size(2)
        x = rearrange(x, 'B C T H W -> (B T) C H W')
        appearance_guidance = repeat(appearance_guidance, 'B C H W -> (B T) C H W', T=T)

        x = self.conv_linear(torch.cat([x, appearance_guidance], dim=1))
        x = self.conv_layer(x)
        x = rearrange(x, '(B T) C H W -> B C T H W', B=B)
        return x


class DoubleConv(nn.Module):
    """(convolution => [GN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 16, mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(mid_channels // 16, mid_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, guidance_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels - guidance_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
        self.cconv = nn.Conv2d(in_channels - guidance_channels, guidance_channels, kernel_size=1, stride=1, padding=0)
        self.sconv = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
    def forward(self, x, guidance=None):
        x = self.up(x)

        if guidance is not None:
            T = x.size(0) // guidance.size(0)
            attention = x.view(guidance.size(0),T,x.size(1),guidance.size(2),guidance.size(3)).mean(dim=1)
            channel_attention = attention.mean(dim=[2,3],keepdim=True)
            spatial_attention = attention.mean(dim=1,keepdim=True)
            
            channel_attention = self.cconv(channel_attention)
            spatial_attention = self.sconv(spatial_attention)

            guidance = channel_attention*guidance + spatial_attention*guidance + guidance
            
            guidance = repeat(guidance, "B C H W -> (B T) C H W", T=T)
            x = torch.cat([x, guidance], dim=1)
        return self.conv(x)


class Aggregator(nn.Module):
    def __init__(self, 
        text_guidance_dim=512,
        text_guidance_proj_dim=128,
        appearance_guidance_dim=512,
        appearance_guidance_proj_dim=128,
        decoder_dims = (64, 32),
        decoder_guidance_dims=(256, 128),
        decoder_guidance_proj_dims=(32, 16),
        num_layers=4,
        nheads=4, 
        hidden_dim=128,
        pooling_size=(6, 6),
        feature_resolution=(24, 24),
        window_size=12,
        attention_type='linear',
        prompt_channel=1,
        pad_len=256,
    ) -> None:
        """
        Cost Aggregation Model for CAT-Seg
        Args:
            text_guidance_dim: Dimension of text guidance
            text_guidance_proj_dim: Dimension of projected text guidance
            appearance_guidance_dim: Dimension of appearance guidance
            appearance_guidance_proj_dim: Dimension of projected appearance guidance
            decoder_dims: Upsampling decoder dimensions
            decoder_guidance_dims: Upsampling decoder guidance dimensions
            decoder_guidance_proj_dims: Upsampling decoder guidance projected dimensions
            num_layers: Number of layers for the aggregator
            nheads: Number of attention heads
            hidden_dim: Hidden dimension for transformer blocks
            pooling_size: Pooling size for the class aggregation layer
                          To reduce computation, we apply pooling in class aggregation blocks to reduce the number of tokens during training
            feature_resolution: Feature resolution for spatial aggregation
            window_size: Window size for Swin block in spatial aggregation
            attention_type: Attention type for the class aggregation. 
            prompt_channel: Number of prompts for ensembling text features. Default: 1
            pad_len: Padding length for the class aggregation. Default: 256
                     pad_len enforces the class aggregation block to have a fixed length of tokens for all inputs
                     This means it either pads the sequence with learnable tokens in class aggregation,
                     or truncates the classes with the initial CLIP cosine-similarity scores.
                     Set pad_len to 0 to disable this feature.
            """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.layers = nn.ModuleList([
            AggregatorLayer(
                hidden_dim=hidden_dim, text_guidance_dim=text_guidance_proj_dim, appearance_guidance=appearance_guidance_proj_dim, 
                nheads=nheads, input_resolution=feature_resolution, pooling_size=pooling_size, window_size=window_size, attention_type=attention_type, pad_len=pad_len,
            ) for _ in range(num_layers)
        ])

        self.conv1 = nn.Conv2d(prompt_channel*4, hidden_dim, kernel_size=7, stride=1, padding=3)

        self.guidance_projection = nn.Sequential(
            nn.Conv2d(appearance_guidance_dim, appearance_guidance_proj_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        ) if appearance_guidance_dim > 0 else None
        
        self.text_guidance_projection = nn.Sequential(
            nn.Linear(text_guidance_dim, text_guidance_proj_dim),
            nn.ReLU(),
        ) if text_guidance_dim > 0 else None

        self.decoder_guidance_projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d, dp, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for d, dp in zip(decoder_guidance_dims, decoder_guidance_proj_dims)
        ]) if decoder_guidance_dims[0] > 0 else None

        self.decoder1 = Up(hidden_dim, decoder_dims[0], decoder_guidance_proj_dims[0])
        self.decoder2 = Up(decoder_dims[0], decoder_dims[1], decoder_guidance_proj_dims[1])
        self.head = nn.Conv2d(decoder_dims[1], 1, kernel_size=3, stride=1, padding=1)

        self.pad_len = pad_len

    def feature_map(self, img_feats, text_feats):
        # concatenated feature volume for feature aggregation baselines
        img_feats = F.normalize(img_feats, dim=1) # B C H W
        img_feats = repeat(img_feats, "B C H W -> B C T H W", T=text_feats.shape[1])
        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        text_feats = text_feats.mean(dim=-2) # average text features over different prompts
        text_feats = F.normalize(text_feats, dim=-1) # B T C
        text_feats = repeat(text_feats, "B T C -> B C T H W", H=img_feats.shape[-2], W=img_feats.shape[-1])
        return torch.cat((img_feats, text_feats), dim=1) # B 2C T H W

    def correlation(self, img_feats, text_feats):
        img_feats0 = F.normalize(img_feats[0], dim=1) # B C H W
        img_feats1 = F.normalize(img_feats[1], dim=1) # B C H W
        img_feats2 = F.normalize(img_feats[2], dim=1) # B C H W
        img_feats3 = F.normalize(img_feats[3], dim=1) # B C H W
        img_feats1= torch.rot90(img_feats1, k=3, dims=(2, 3))
        img_feats2 = torch.rot90(img_feats2, k=2, dims=(2, 3))
        img_feats3 = torch.rot90(img_feats3, k=1, dims=(2, 3))

        img_feats = torch.cat((img_feats0.unsqueeze(dim=1),img_feats1.unsqueeze(dim=1),img_feats2.unsqueeze(dim=1),img_feats3.unsqueeze(dim=1)),dim=1)

        text_feats = F.normalize(text_feats, dim=-1) # B T P C
        corr = torch.einsum('bnchw, btpc -> bnpthw', img_feats, text_feats)
        corr = rearrange(corr, 'B N P T H W -> B (N P) T H W')
        return corr

    def corr_embed(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B P T H W -> (B T) P H W')
        corr_embed = self.conv1(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed
    
    def corr_projection(self, x, proj):
        corr_embed = rearrange(x, 'B C T H W -> B T H W C')
        corr_embed = proj(corr_embed)
        corr_embed = rearrange(corr_embed, 'B T H W C -> B C T H W')
        return corr_embed

    def upsample(self, x):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = F.interpolate(corr_embed, scale_factor=2, mode='bilinear', align_corners=True)
        corr_embed = rearrange(corr_embed, '(B T) C H W -> B C T H W', B=B)
        return corr_embed

    def conv_decoder(self, x, guidance):
        B = x.shape[0]
        corr_embed = rearrange(x, 'B C T H W -> (B T) C H W')
        corr_embed = self.decoder1(corr_embed, guidance[0])
        corr_embed = self.decoder2(corr_embed, guidance[1])
        corr_embed = self.head(corr_embed)
        corr_embed = rearrange(corr_embed, '(B T) () H W -> B T H W', B=B)
        return corr_embed
    
    def forward(self, img_feats, text_feats, appearance_guidance):
        """
        Arguments:
            img_feats: (B, C, H, W)
            text_feats: (B, T, P, C)
            apperance_guidance: tuple of (B, C, H, W)
        """
        classes = None

        corr = self.correlation(img_feats, text_feats)
        if self.pad_len > 0 and text_feats.size(1) > self.pad_len: #B (N P) T H W 
            avg = corr.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0]  #B  T (N P) H W 
            classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
            th_text = F.normalize(text_feats, dim=-1)
            th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
            orig_clases = text_feats.size(1)
            # img_feats = F.normalize(img_feats, dim=1) # B C H W
            img_feats0 = F.normalize(img_feats[0], dim=1) # B C H W
            img_feats1 = F.normalize(img_feats[1], dim=1) # B C H W
            img_feats2 = F.normalize(img_feats[2], dim=1) # B C H W
            img_feats3 = F.normalize(img_feats[3], dim=1) # B C H W
            img_feats1= torch.rot90(img_feats1, k=3, dims=(2, 3))
            img_feats2 = torch.rot90(img_feats2, k=2, dims=(2, 3))
            img_feats3 = torch.rot90(img_feats3, k=1, dims=(2, 3))

            img_feats = torch.cat((img_feats0.unsqueeze(dim=1),img_feats1.unsqueeze(dim=1),img_feats2.unsqueeze(dim=1),img_feats3.unsqueeze(dim=1)),dim=1)
            text_feats = th_text
            corr = torch.einsum('bnchw, btpc -> bnpthw', img_feats, text_feats)
            corr = rearrange(corr, 'B N P T H W -> B (N P) T H W')

            # text_feats = th_text
            # corr1 = torch.einsum('bchw, btpc -> bpthw', img_feats, th_text)

        # corr2 = self.correlation(img_feats[1], text_feats)
        # if self.pad_len > 0 and text_feats.size(1) > self.pad_len:
        #     avg = corr2.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
        #     classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
        #     th_text = F.normalize(text_feats, dim=-1)
        #     th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
        #     orig_clases = text_feats.size(1)
        #     img_feats = F.normalize(img_feats, dim=1) # B C H W
        #     text_feats = th_text
        #     corr2 = torch.einsum('bchw, btpc -> bpthw', img_feats, th_text)

        # corr3 = self.correlation(img_feats[2], text_feats)
        # if self.pad_len > 0 and text_feats.size(1) > self.pad_len:
        #     avg = corr3.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
        #     classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
        #     th_text = F.normalize(text_feats, dim=-1)
        #     th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
        #     orig_clases = text_feats.size(1)
        #     img_feats = F.normalize(img_feats, dim=1) # B C H W
        #     text_feats = th_text
        #     corr3 = torch.einsum('bchw, btpc -> bpthw', img_feats, th_text)

        # corr4 = self.correlation(img_feats[3], text_feats)
        # if self.pad_len > 0 and text_feats.size(1) > self.pad_len:
        #     avg = corr4.permute(0, 2, 1, 3, 4).flatten(-3).max(dim=-1)[0] 
        #     classes = avg.topk(self.pad_len, dim=-1, sorted=False)[1]
        #     th_text = F.normalize(text_feats, dim=-1)
        #     th_text = torch.gather(th_text, dim=1, index=classes[..., None, None].expand(-1, -1, th_text.size(-2), th_text.size(-1)))
        #     orig_clases = text_feats.size(1)
        #     img_feats = F.normalize(img_feats, dim=1) # B C H W
        #     text_feats = th_text
        #     corr4 = torch.einsum('bchw, btpc -> bpthw', img_feats, th_text)
    

        corr_embed = self.corr_embed(corr)

        projected_guidance, projected_text_guidance, projected_decoder_guidance = None, None, [None, None]
        if self.guidance_projection is not None:
            projected_guidance = self.guidance_projection(appearance_guidance[0])
        if self.decoder_guidance_projection is not None:
            projected_decoder_guidance = [proj(g) for proj, g in zip(self.decoder_guidance_projection, appearance_guidance[1:])]

        if self.text_guidance_projection is not None:
            text_feats = text_feats.mean(dim=-2)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            projected_text_guidance = self.text_guidance_projection(text_feats)

        for layer in self.layers:
            corr_embed = layer(corr_embed, projected_guidance, projected_text_guidance)

        logit = self.conv_decoder(corr_embed, projected_decoder_guidance)
        if classes is not None:
            out = torch.full((logit.size(0), orig_clases, logit.size(2), logit.size(3)), -100., device=logit.device)
            out.scatter_(dim=1, index=classes[..., None, None].expand(-1, -1, logit.size(-2), logit.size(-1)), src=logit)
            logit = out
        return logit
