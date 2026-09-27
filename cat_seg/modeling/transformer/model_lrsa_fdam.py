import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from timm.layers import PatchEmbed, Mlp, DropPath, to_2tuple, to_ntuple, trunc_normal_, _assert

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
# ---------------------------
# FDAM: StarReLU (原样)
# ---------------------------
class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias  = nn.Parameter(bias_value  * torch.ones(1), requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


# ---------------------------
# FDAM: Mlp (给 GroupDynamicScale 的 reweight 用；保持原逻辑)
# ---------------------------
class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0., bias=False):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# ---------------------------
# FDAM: GroupDynamicScale (原样逻辑 + dtype 回写，避免混精度类型问题)
# ---------------------------
class GroupDynamicScale(nn.Module):
    def __init__(
        self,
        dim,
        expansion_ratio=1,
        reweight_expansion_ratio=.125,
        act1_layer=StarReLU,
        act2_layer=nn.Identity,
        bias=False,
        num_filters=4,
        size=14,
        weight_resize=True,
        group=16,
        init_scale=1e-5,
    ):
        super().__init__()
        self.size = int(size)
        self.filter_size = self.size // 2 + 1
        self.num_filters = int(num_filters)
        self.dim = int(dim)
        self.med_channels = int(expansion_ratio * dim)
        self.weight_resize = bool(weight_resize)
        self.group = int(group)
        assert self.dim % self.group == 0, f"dim({dim}) must be divisible by group({group})"

        # routing: (B, dim) -> (B, group*num_filters) -> reshape (B, group, num_filters)
        self.reweight = Mlp(dim, reweight_expansion_ratio, self.group * self.num_filters, bias=False)
        self.complex_weights = nn.Parameter(
            torch.randn(self.num_filters, self.dim // self.group, self.size, self.filter_size, dtype=torch.float32) * init_scale
        )
        trunc_normal_(self.complex_weights, std=init_scale)
        self.act2 = act2_layer()

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        B, C, H, W = x.shape
        in_dtype = x.dtype

        x_rfft = torch.fft.rfft2(x.to(torch.float32), dim=(2, 3), norm='ortho')  # (B,C,RH,RW) complex
        _, _, RH, RW = x_rfft.shape

        # routing uses spatial mean on channels_last in FDAM；等价地用 x.mean(H,W) 即可
        x_mean = x.to(torch.float32).mean(dim=(2, 3))  # (B,C)
        routeing = self.reweight(x_mean).view(B, -1, self.num_filters).tanh_()  # (B, group, num_filters)

        weight = self.complex_weights  # (F, Cg, size, size//2+1)
        if self.weight_resize and (weight.shape[2] != RH or weight.shape[3] != RW):
            weight = F.interpolate(weight, size=(RH, RW), mode='bicubic', align_corners=True)

        # einsum: (B,G,F) x (F,Cg,RH,RW) -> (B,G,Cg,RH,RW) -> reshape (B,C,RH,RW)
        weight = torch.einsum('bgf,fchw->bgchw', routeing, weight).reshape(B, C, RH, RW)

        x_rfft = torch.view_as_complex(torch.stack([x_rfft.real * weight, x_rfft.imag * weight], dim=-1))
        x_out = torch.fft.irfft2(x_rfft, s=(H, W), dim=(2, 3), norm='ortho')
        return x_out.to(in_dtype)

def nlc_to_nchw(x, hw_shape):
    H, W = hw_shape
    B, L, C = x.shape
    assert L == H * W, f"seq_len {L} != H*W {H*W}"
    return x.transpose(1, 2).reshape(B, C, H, W)

def nchw_to_nlc(x):
    return x.flatten(2).transpose(1, 2).contiguous()

class GuidedPoolingAttention(nn.Module):
    def __init__(
        self,
        dim,
        guidance_dim,
        num_heads=2,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        pooled_sizes=(8, 6, 5, 3),
        q_pooled_size=1,
        q_conv=False,
        kv_mode="pyramid",
        # FDAM AttInv
        use_attinv=False,
        lf_dy_weight=True,
        hf_dy_weight=True,
        attinv_init_gamma=1e-5,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.guidance_dim = int(guidance_dim)
        self.num_heads = int(num_heads)
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim + self.guidance_dim, dim, bias=qkv_bias) if self.guidance_dim > 0 else nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim + self.guidance_dim, dim, bias=qkv_bias) if self.guidance_dim > 0 else nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pooled_sizes = list(pooled_sizes)
        self.norm_x = nn.LayerNorm(dim)
        self.norm_g = nn.LayerNorm(self.guidance_dim) if self.guidance_dim > 0 else None

        self.q_pooled_size = int(q_pooled_size)
        if q_conv and self.q_pooled_size > 1:
            self.q_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
            self.q_norm = nn.LayerNorm(dim)
        else:
            self.q_conv = None
            self.q_norm = None

        self.kv_mode = kv_mode

        # ---- AttInv 固化开关：避免 forward 里“有时用有时不用”导致 DDP unused ----
        self.lf_dy_weight = bool(lf_dy_weight)
        self.hf_dy_weight = bool(hf_dy_weight)
        self.use_attinv = bool(use_attinv)
        self.attinv_active = self.use_attinv and (self.kv_mode == "full") and (self.q_pooled_size <= 1)

        if self.attinv_active:
            self.dy_freq_starrelu = StarReLU()
            self.ignore_cls_token = 0
            if self.lf_dy_weight:
                self.dy_freq_2 = nn.Linear(dim, self.num_heads, bias=True)
                self.lf_gamma = nn.Parameter(attinv_init_gamma * torch.ones((dim)), requires_grad=True)
            if self.hf_dy_weight:
                self.dy_freq = nn.Linear(dim, self.num_heads, bias=True)
                self.hf_gamma = nn.Parameter(attinv_init_gamma * torch.ones((dim)), requires_grad=True)

    def _aspect_pool_size(self, base, H, W):
        eps = 1e-3
        if W >= H:
            return (base, int(round(W * float(base) / H + eps)))
        else:
            return (int(round(H * float(base) / W + eps)), base)

    def forward(self, x, H, W, guidance=None, d_convs=None):
        B, N, C = x.shape
        H, W = int(H), int(W)

        # ----- Q -----
        if self.q_pooled_size > 1:
            q_size = self._aspect_pool_size(self.q_pooled_size, H, W)
            x2d = x.transpose(1, 2).reshape(B, C, H, W)
            q2d = F.adaptive_avg_pool2d(x2d, q_size)
            _, _, Hq, Wq = q2d.shape

            if self.q_conv is not None:
                q2d = q2d + self.q_conv(q2d)
                q_tok = self.q_norm(q2d.flatten(2).transpose(1, 2))
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
            q = self.q(q_in).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
            Lq = N
            Hq, Wq = H, W

        # ----- K/V -----
        if self.kv_mode == "full":
            x_tok = self.norm_x(x)
            if self.guidance_dim > 0:
                g_tok = self.norm_g(guidance)
                k_in = torch.cat([x_tok, g_tok], dim=-1)
            else:
                k_in = x_tok
            k = self.k(k_in).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
            v = self.v(x_tok).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        elif self.kv_mode == "pyramid":
            if d_convs is None:
                d_convs = [nn.Identity() for _ in self.pooled_sizes]

            x2d = x.transpose(1, 2).reshape(B, C, H, W)
            g2d = guidance.transpose(1, 2).reshape(B, self.guidance_dim, H, W) if self.guidance_dim > 0 else None

            pooled_x_list, pooled_g_list = [], []
            for base, conv in zip(self.pooled_sizes, d_convs):
                psize = self._aspect_pool_size(int(base), H, W)
                px = F.adaptive_avg_pool2d(x2d, psize)
                px = px + conv(px)
                pooled_x_list.append(px.flatten(2))
                if self.guidance_dim > 0:
                    pg = F.adaptive_avg_pool2d(g2d, psize)
                    pooled_g_list.append(pg.flatten(2))

            pooled_x = torch.cat(pooled_x_list, dim=2)
            pooled_x_tok = self.norm_x(pooled_x.transpose(1, 2).contiguous())

            if self.guidance_dim > 0:
                pooled_g = torch.cat(pooled_g_list, dim=2)
                pooled_g_tok = self.norm_g(pooled_g.transpose(1, 2).contiguous())
                k_in = torch.cat([pooled_x_tok, pooled_g_tok], dim=-1)
            else:
                k_in = pooled_x_tok

            k = self.k(k_in).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
            v = self.v(pooled_x_tok).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        else:
            raise ValueError(f"Unknown kv_mode={self.kv_mode}")

        # ----- attn -----
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, Lq, C).contiguous()

        # ----- AttInv：只要 active，每一步都执行（避免 DDP unused） -----
        if self.attinv_active:
            # 此时必为 kv_mode=full 且 q_pooled_size=1，因此 Lq=N=H*W，v_tok 与 out 对齐
            dy_feat = self.dy_freq_starrelu(x[:, self.ignore_cls_token:])  # (B,N,C)

            if self.lf_dy_weight:
                dy_lf = self.dy_freq_2(dy_feat).tanh_()  # (B,N,head)
                dy_lf = dy_lf.reshape(B, N, self.num_heads, 1).repeat(1, 1, 1, C // self.num_heads).reshape(B, N, C)

            if self.hf_dy_weight:
                dy = F.softplus(self.dy_freq(dy_feat))
                dy2 = dy ** 2
                dy = 2 * dy2 / (dy2 + 0.3678)
                dy = dy.reshape(B, N, self.num_heads, 1).repeat(1, 1, 1, C // self.num_heads).reshape(B, N, C)

            v_tok = v.permute(0, 2, 1, 3).reshape(B, N, C)
            v_hf = v_tok - out

            if self.lf_dy_weight:
                out = out + out * dy_lf * self.lf_gamma.view(1, 1, -1)
            if self.hf_dy_weight:
                out = out + dy * v_hf * self.hf_gamma.view(1, 1, -1)

        out = self.proj_drop(self.proj(out))

        # Q pooled -> upsample back
        if self.q_pooled_size > 1:
            out2d = out.transpose(1, 2).reshape(B, C, Hq, Wq)
            out2d = F.interpolate(out2d, size=(H, W), mode="bilinear", align_corners=True)
            out = out2d.flatten(2).transpose(1, 2).contiguous()

        return out


# ---------------------------
# LRSARefineBlock：按 deit_fdam.py 的方式加 gamma_1/gamma_2 + GroupDynamicScale(Attn/MLP)
# ---------------------------
def _pick_group(dim, preferred=16):
    # 选一个 <=preferred 的最大约数，保证 dim % group == 0
    preferred = min(int(preferred), int(dim))
    for g in range(preferred, 0, -1):
        if dim % g == 0:
            return g
    return 1

class LRSARefineBlock(nn.Module):
    def __init__(
        self,
        dim,
        guidance_dim,
        num_heads,
        pooled_sizes=(8, 6, 5, 3),
        q_pooled_size=1,
        q_conv=False,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        extra_act=False,
        use_prenorm=False,
        kv_mode="pyramid",
        use_attinv=False,
        use_fdam_freqscale=False,
        fdam_num_filters=4,
        fdam_group_preferred=16,
        fdam_init_scale=1e-5,
        fdam_base_size=24,          # <<< 新增：FDAM 的 base token size（训练分辨率的 token H）
        init_values=1e-4,
    ):
        super().__init__()
        self.kv_mode = kv_mode

        self.pre_norm = nn.BatchNorm2d(dim) if use_prenorm else nn.Identity()
        self.cpe = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

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
            kv_mode=kv_mode,
            use_attinv=use_attinv,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = IRB(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=drop,
            ksize=3,
            extra_act=extra_act
        )

        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)

        # 关键：只有 pyramid 才注册 d_convs，full 不注册，避免 DDP unused 参数
        if self.kv_mode == "pyramid":
            self.d_convs = nn.ModuleList([
                nn.Conv2d(dim, dim, 3, 1, 1, groups=dim) for _ in pooled_sizes
            ])
        else:
            self.d_convs = None

        self.use_fdam_freqscale = bool(use_fdam_freqscale)
        if self.use_fdam_freqscale:
            group = _pick_group(dim, fdam_group_preferred)
            self.freq_scale_1 = GroupDynamicScale(
                dim=dim, group=group, num_filters=fdam_num_filters,
                size=int(fdam_base_size), init_scale=fdam_init_scale, weight_resize=True
            )
            self.freq_scale_2 = GroupDynamicScale(
                dim=dim, group=group, num_filters=fdam_num_filters,
                size=int(fdam_base_size), init_scale=fdam_init_scale, weight_resize=True
            )
        else:
            self.freq_scale_1 = None
            self.freq_scale_2 = None

    def forward(self, x, H, W, guidance=None):
        B, N, C = x.shape

        # CPE
        x2d = x.transpose(1, 2).reshape(B, C, H, W)
        x2d = self.pre_norm(x2d)
        x2d = self.cpe(x2d) + x2d
        x = x2d.flatten(2).transpose(1, 2).contiguous()

        # Attention
        d_convs = self.d_convs if self.kv_mode == "pyramid" else None
        x_att = self.attn(self.norm1(x), H, W, guidance=guidance, d_convs=d_convs)

        if self.use_fdam_freqscale:
            x_att_2d = nlc_to_nchw(x_att, (H, W))
            x_att_2d = self.freq_scale_1(x_att_2d) + x_att_2d
            x_att = nchw_to_nlc(x_att_2d)

        x = x + self.drop_path(self.gamma_1 * x_att)

        # MLP/IRB
        x_mlp = self.mlp(self.norm2(x), H, W)

        if self.use_fdam_freqscale:
            x_mlp_2d = nlc_to_nchw(x_mlp, (H, W))
            x_mlp_2d = self.freq_scale_2(x_mlp_2d) + x_mlp_2d
            x_mlp = nchw_to_nlc(x_mlp_2d)

        x = x + self.drop_path(self.gamma_2 * x_mlp)
        return x

# ---------------------------
# SwinTransformerBlockWrapper：block1(平滑) + block2(FDAM细节回注)
# ---------------------------

class SwinTransformerBlockWrapper(nn.Module):
    """
    目标实现：
    - 两层 q_pooled_size 都为 12
    - Block1: pyramid KV + 更细粒度 pooled_sizes（包含 12 档）
    - Block2: full KV + FDAM（只在 Block2 开启）
    """
    def __init__(
        self,
        dim,
        appearance_guidance_dim,
        input_resolution,
        nheads=4,
        q_pooled_size=12,                     # <<< 固定 q=12（可外部配）
        pooled_sizes=(12, 10, 8, 6, 5, 3),    # <<< 细粒度档位（你要的第二件事）
        enable_fdam_in_block2=True,
        fdam_num_filters=4,
        fdam_group_preferred=16,
        fdam_init_scale=1e-5,
    ):
        super().__init__()
        H, W = to_2tuple(input_resolution)

        # q=12，但别超过 token 分辨率
        q = min(int(q_pooled_size), max(int(H), int(W)))
        q = max(q, 1)

        # pooled_sizes 也别超过 token 分辨率；去重、过滤、排序（从大到小）
        max_hw = max(int(H), int(W))
        ps = [int(s) for s in pooled_sizes]
        ps = [min(max(s, 1), max_hw) for s in ps]
        ps = sorted(set(ps), reverse=True)

        self.guidance_norm = nn.LayerNorm(appearance_guidance_dim) if appearance_guidance_dim > 0 else None

        # Block1：pyramid KV + q=12（你高分辨率收益的关键结构）
        self.block_1 = LRSARefineBlock(
            dim=dim,
            guidance_dim=appearance_guidance_dim,
            num_heads=nheads,
            pooled_sizes=ps,
            q_pooled_size=q,
            q_conv=False,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            use_prenorm=False,
            kv_mode="pyramid",
            use_attinv=False,
            use_fdam_freqscale=False,          # 建议先别在 block1 叠 FDAM
        )

        # Block2：full KV + q=12 + FDAM（补细节/补低分辨率）
        self.block_2 = LRSARefineBlock(
            dim=dim,
            guidance_dim=appearance_guidance_dim,
            num_heads=nheads,
            pooled_sizes=ps,                   # full KV 不用它，但保留接口一致
            q_pooled_size=q,
            q_conv=True,                       # q>1 时 q_conv 才生效
            mlp_ratio=4.0,
            qkv_bias=True,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            norm_layer=nn.LayerNorm,
            use_prenorm=False,
            kv_mode="full",                    # <<< 关键：Block2 full KV
            use_attinv=False,                  # q=12 时 AttInv 不会激活（你的实现也不建议强开）
            use_fdam_freqscale=bool(enable_fdam_in_block2),
            fdam_num_filters=fdam_num_filters,
            fdam_group_preferred=fdam_group_preferred,
            fdam_init_scale=fdam_init_scale,
            fdam_base_size=max_hw,             # <<< 关键：FDAM base size 跟 token H/W 对齐
        )

        self.input_resolution = (int(H), int(W))

    def forward(self, x, appearance_guidance):
        """
        x: B C T H W
        appearance_guidance: B Cg H W
        """
        B, C, T, H, W = x.shape
        assert (H, W) == self.input_resolution, f"Expected {self.input_resolution}, got {(H, W)}"

        x = rearrange(x, "B C T H W -> (B T) (H W) C")

        g = None
        if appearance_guidance is not None:
            g = repeat(appearance_guidance, "B C H W -> (B T) (H W) C", T=T)
            if self.guidance_norm is not None:
                g = self.guidance_norm(g)

        x = self.block_1(x, H, W, guidance=g)
        x = self.block_2(x, H, W, guidance=g)

        x = rearrange(x, "(B T) (H W) C -> B C T H W", B=B, T=T, H=H, W=W)
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
