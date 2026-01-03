class DropPath(nn.Module):
    """Stochastic Depth (per sample)."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.size(0),) + (1,) * (x.ndim - 1)
        rnd = x.new_empty(shape).bernoulli_(keep)
        return x * rnd / keep


class LayerScale(nn.Module):
    """Per-channel residual scaling."""
    def __init__(self, channels, init_value=1e-3):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1) * init_value)

    def forward(self, x):
        return x * self.gamma


class GRN2d(nn.Module):
    """Global Response Normalization (ConvNeXt-V2 style) for NCHW tensors."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        gx = torch.sqrt((x ** 2).mean(dim=(2, 3), keepdim=True) + self.eps)
        nx = x / (gx + self.eps)
        return x + self.gamma * nx + self.beta


# =============== CoordAttention (lightweight, long-range along H/W) ===============
class CoordAttention(nn.Module):
    """
    Coordinate Attention (Hou et al.):
    Compresses features along H and W separately and recovers attention maps.
    """
    def __init__(self, channels, reduction=32):
        super().__init__()
        m = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, m, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(m)
        self.act = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(m, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(m, channels, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        x_h = F.adaptive_avg_pool2d(x, (H, 1))          # (B, C, H, 1)
        x_w = F.adaptive_avg_pool2d(x, (1, W))          # (B, C, 1, W)
        x_w = x_w.permute(0, 1, 3, 2)                   # (B, C, W, 1)
        y = torch.cat([x_h, x_w], dim=2)                # (B, C, H+W, 1)
        y = self.act(self.bn1(self.conv1(y)))
        y_h, y_w = torch.split(y, [H, W], dim=2)
        a_h = torch.sigmoid(self.conv_h(y_h))           # (B, C, H, 1)
        a_w = torch.sigmoid(self.conv_w(y_w).permute(0, 1, 3, 2))  # (B, C, 1, W)
        return x * a_h * a_w


# =============== 1) TinyGateV2: SwiGLU + residual gate + GRN ===============
class TinyGateV2(nn.Module):
    """
    Enhanced TinyGate:
    LN -> SwiGLU (Linear 2C) -> Linear C -> sigmoid,
    combined with GRN and residual scaling for stable and sharper gating.
    """
    def __init__(self, channels, hidden=None, layerscale=1e-3):
        super().__init__()
        if hidden is None:
            hidden = max(8, channels // 3)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.fc_in = nn.Linear(channels, hidden * 2, bias=False)  # SwiGLU
        self.fc_out = nn.Linear(hidden, channels, bias=False)
        self.grn = GRN2d(channels)
        self.ls = LayerScale(channels, init_value=layerscale)

    def forward(self, x):
        B, C, _, _ = x.shape
        v = F.adaptive_avg_pool2d(x, 1).view(B, C)
        v = self.norm(v)
        a, b = self.fc_in(v).chunk(2, dim=-1)
        v = a * F.silu(b)                               # SwiGLU
        gate = torch.sigmoid(self.fc_out(v)).view(B, C, 1, 1)
        y = x * gate
        y = self.grn(y)
        return x + self.ls(y - x)                       # residual gated update


# =============== 2) CrossLayerChannelFusionV2: token mixing + GEGLU + FiLM ===============
class CrossLayerChannelFusionV2(nn.Module):
    """
    - PreNorm (LayerNorm over channels in NHWC format)
    - Token mixer: depthwise 3x3 convolution (local context) + GRN
    - Gated MLP: GEGLU (Linear 2H) -> Linear C
    - FiLM modulation: scale and bias generated from global average pooling
    """
    def __init__(self, channels, expansion_factor=2):
        super().__init__()
        self.channels = channels
        hidden = channels * expansion_factor

        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.grn = GRN2d(channels)

        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.glu = nn.Linear(channels, hidden * 2, bias=False)
        self.proj = nn.Linear(hidden, channels, bias=False)

        self.film = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, max(16, channels // 2)), nn.SiLU(inplace=True),
            nn.Linear(max(16, channels // 2), channels * 2)  # (scale, bias)
        )

    def forward(self, x):
        # token mixer
        xm = self.grn(self.dw(x))

        # gated MLP per pixel (operate on last dimension)
        y = x.permute(0, 2, 3, 1)                        # (B, H, W, C)
        y = self.norm(y)
        a, b = self.glu(y).chunk(2, dim=-1)              # GEGLU
        y = F.gelu(a) * b
        y = self.proj(y)
        y = y.permute(0, 3, 1, 2).contiguous()           # (B, C, H, W)

        # FiLM modulation from global average pooling
        s, b = self.film(x).chunk(2, dim=-1)
        s = torch.sigmoid(s).view(x.size(0), self.channels, 1, 1)
        b = b.view(x.size(0), self.channels, 1, 1)

        return (xm + y) * s + b


# =============== 3) PAUG: SK-softmax + CoordAtt + DropPath ===============
class PAUG_PanAxisUnifiedGating(nn.Module):
    def __init__(self, channels, drop_path=0.0):
        super().__init__()
        self.C = channels

        # Channel branch
        self.channel_branch = CrossLayerChannelFusionV2(channels)

        # Spatial selective kernels
        self.dw1x3 = nn.Conv2d(channels, channels, kernel_size=(1, 3), padding=(0, 1),
                               groups=channels, bias=False)
        self.dw3x1 = nn.Conv2d(channels, channels, kernel_size=(3, 1), padding=(1, 0),
                               groups=channels, bias=False)
        self.dw3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                               groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

        # Softmax router + CoordAttention modulation
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, max(4, channels // 4)), nn.SiLU(inplace=True),
            nn.Linear(max(4, channels // 4), 3)
        )
        self.coord = CoordAttention(channels, reduction=32)

        # Final gate: TinyGateV2 + residual tools
        self.gate = TinyGateV2(channels, hidden=max(8, channels // 2))
        self.ls = LayerScale(channels, init_value=1e-3)
        self.dp = DropPath(drop_path)

    def forward(self, x):
        identity = x

        # Channel branch
        ch = self.channel_branch(x)

        # Spatial branch with SK-softmax + CoordAttention
        b1 = self.dw1x3(x)
        b2 = self.dw3x1(x)
        b3 = self.dw3x3(x)
        w = torch.softmax(self.router(x), dim=-1)        # (B, 3)
        w1, w2, w3 = [w[:, i].view(-1, 1, 1, 1) for i in range(3)]
        sp = w1 * b1 + w2 * b2 + w3 * b3
        sp = self.act(self.bn(self.pw(sp)))
        sp = self.coord(sp)                              # spatial long-range guidance

        fused = ch + sp
        fused = self.gate(fused)                         # stronger gating
        out = identity + self.dp(self.ls(fused))         # residual + layerscale + stochastic depth
        return out

# =============== 4) FCSAV2: multi-scale with softmax router + scale-drop ===============
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==== per-scale unit (lightweight, stable) ====
class _DS(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.SiLU(inplace=True),
        )
        # If TinyGateV2 & GRN2d are available, they can be added here for stronger modeling.
        self.gate = nn.Identity()

    def forward(self, x):
        return self.gate(self.block(x))


# ==== head for offset + attention prediction ====
class OffsetAttnHead(nn.Module):
    """
    Predicts sampling offsets and attention weights for K points
    from the full-resolution query feature.
    """
    def __init__(self, channels, K=4, hidden_ratio=0.5):
        super().__init__()
        h = max(4, int(channels * hidden_ratio))
        self.conv = nn.Sequential(
            nn.Conv2d(channels, h, 3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(h, K * 3, 1, bias=True)  # (K*2 offsets + K weights)
        )
        self.K = K

    def forward(self, q):                   # q: (B, C, Hf, Wf)
        B, C, H, W = q.shape
        out = self.conv(q)                  # (B, K*3, H, W)
        off, w = torch.split(out, [self.K * 2, self.K], dim=1)
        off = off.view(B, self.K, 2, H, W)  # (B, K, 2, H, W)
        w = F.softmax(w.view(B, self.K, H, W), dim=1)  # (B, K, H, W)
        return off, w


# ==== utility to generate base grid (fine -> coarse, normalized to [-1, 1]) ====
def make_base_grid(Hf, Wf, Hc, Wc, device, dtype):
    yy, xx = torch.meshgrid(
        torch.arange(Hf, device=device, dtype=dtype),
        torch.arange(Wf, device=device, dtype=dtype),
        indexing='ij'
    )
    xc = (xx + 0.5) * (Wc / Wf) - 0.5
    yc = (yy + 0.5) * (Hc / Hf) - 0.5
    grid_x = 2.0 * xc / max(Wc - 1, 1) - 1.0
    grid_y = 2.0 * yc / max(Hc - 1, 1) - 1.0
    return torch.stack((grid_x, grid_y), dim=-1)  # (Hf, Wf, 2)


# ==== deformable sampling: coarse -> fine ====
def deform_sample(coarse, off, w, base_grid, r_max=2.0, align_corners=True):
    """
    coarse: (B, C, Hc, Wc)
    off:    (B, K, 2, Hf, Wf)
    w:      (B, K, Hf, Wf)
    base_grid: (Hf, Wf, 2), normalized w.r.t. coarse feature map
    r_max: maximum offset radius (in coarse pixels)
    """
    B, C, Hc, Wc = coarse.shape
    B2, K, _, Hf, Wf = off.shape
    assert B == B2

    step_x = 2.0 / max(Wc - 1, 1)  # one coarse pixel in normalized coordinates
    step_y = 2.0 / max(Hc - 1, 1)

    dx = torch.tanh(off[:, :, 0]) * (r_max * step_x)
    dy = torch.tanh(off[:, :, 1]) * (r_max * step_y)
    delta = torch.stack((dx, dy), dim=-1)            # (B, K, Hf, Wf, 2)

    base = base_grid.to(coarse.device, coarse.dtype).unsqueeze(0).unsqueeze(0)
    grid = (base + delta).reshape(B * K, Hf, Wf, 2)

    src = coarse.unsqueeze(1).expand(B, K, C, Hc, Wc).reshape(B * K, C, Hc, Wc)
    samp = F.grid_sample(
        src, grid, mode='bilinear',
        padding_mode='border',
        align_corners=align_corners
    )
    samp = samp.view(B, K, C, Hf, Wf)

    w = w.unsqueeze(2)                              # (B, K, 1, Hf, Wf)
    return (w * samp).sum(dim=1)                    # (B, C, Hf, Wf)


# ==== Cross-Scale Deformable Mixer ====
class CSDM(nn.Module):
    """
    Cross-Scale Deformable Mixer:
      - Generates three scales: full (s0), half (s1), and quarter (s2)
      - Applies lightweight processing at each scale (_DS)
      - Learns offset + attention to sample from s1 and s2 to full resolution
      - Uses a softmax router over three branches (z0, y1, y2)
      - Final 1x1 projection with residual connection
    """
    def __init__(self, channels, K=4, r_max=2.0,
                 head_hidden_ratio=0.5, router_hidden=None):
        super().__init__()
        C = channels
        self.proc0 = _DS(C)
        self.proc1 = _DS(C)
        self.proc2 = _DS(C)

        # Query projection (kept lightweight with same channel size)
        self.qproj = nn.Conv2d(C, C, 1, bias=False)

        # Separate offset-attention heads for s1 and s2
        self.head1 = OffsetAttnHead(C, K=K, hidden_ratio=head_hidden_ratio)
        self.head2 = OffsetAttnHead(C, K=K, hidden_ratio=head_hidden_ratio)

        hrouter = router_hidden or max(16, C)
        self.router = nn.Sequential(
            nn.Linear(C * 3, hrouter), nn.SiLU(inplace=True),
            nn.Linear(hrouter, 3)
        )
        self.final = nn.Conv2d(C, C, 1, bias=False)

        self.K = K
        self.r_max = float(r_max)

    def forward(self, x):
        B, C, H, W = x.shape

        # Multi-scale features
        s0 = x
        s1 = F.adaptive_avg_pool2d(x, (max(1, H // 2), max(1, W // 2)))
        s2 = F.adaptive_avg_pool2d(x, (max(1, H // 4), max(1, W // 4)))

        # Per-scale processing
        z0 = self.proc0(s0)                 # (B, C, H, W)
        z1 = self.proc1(s1)                 # (B, C, H/2, W/2)
        z2 = self.proc2(s2)                 # (B, C, H/4, W/4)

        # Full-resolution query for offset/attention prediction
        q = self.qproj(z0)

        # Offsets and attention weights
        off1, w1 = self.head1(q)
        off2, w2 = self.head2(q)

        # Base grids for coarse-to-fine sampling
        base1 = make_base_grid(H, W, z1.size(2), z1.size(3), q.device, q.dtype)
        base2 = make_base_grid(H, W, z2.size(2), z2.size(3), q.device, q.dtype)

        # Deformable sampling from z1 and z2 to full resolution
        y1 = deform_sample(z1, off1, w1, base1, r_max=self.r_max)
        y2 = deform_sample(z2, off2, w2, base2, r_max=self.r_max)

        # Softmax router over three branches (z0, y1, y2)
        d0 = F.adaptive_avg_pool2d(z0, 1).view(B, C)
        d1 = F.adaptive_avg_pool2d(y1, 1).view(B, C)
        d2 = F.adaptive_avg_pool2d(y2, 1).view(B, C)
        alpha = torch.softmax(
            self.router(torch.cat([d0, d1, d2], dim=-1)),
            dim=-1
        )
        a0, a1, a2 = [alpha[:, i].view(B, 1, 1, 1) for i in range(3)]

        out = a0 * z0 + a1 * y1 + a2 * y2
        out = self.final(out)
        return x + out


class CoDeX_Net(nn.Module):
    def __init__(self,
                 channels_per_stage=(8, 8, 16, 16, 32, 32),
                 num_classes=12,
                 drop_path_rate=0.05):
        super().__init__()
        assert len(channels_per_stage) == 6
        self.chs = list(channels_per_stage)
        c0 = self.chs[0]

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, c0, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c0),
            nn.SiLU(inplace=True),
            nn.Conv2d(c0, c0, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c0),
            nn.SiLU(inplace=True),
        )

        # DropPath schedule for 6 blocks
        total = len(self.chs)
        dprs = torch.linspace(0, drop_path_rate, steps=total).tolist()

        self.blocks = nn.ModuleList()
        for i, ch in enumerate(self.chs):
            in_ch = self.chs[i - 1] if i > 0 else c0
            proj = nn.Conv2d(in_ch, ch, 1, bias=False) if in_ch != ch else None
            module = (
                PAUG_PanAxisUnifiedGating(ch, drop_path=dprs[i])
                if (i % 2 == 0)
                else CSDM(channels=ch, K=4, r_max=2.0)
            )
            self.blocks.append(nn.ModuleDict({
                "proj": proj,
                "module": module
            }))

        final_ch = self.chs[-1]
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_ch, max(128, final_ch * 4)),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(max(128, final_ch * 4), num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        for blk in self.blocks:
            if blk["proj"] is not None:
                x = blk["proj"](x)
            x = blk["module"](x)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.classifier(x)


channels = (8, 8, 16, 16, 32, 32)
model = CoDeX_Net(
    channels_per_stage=channels,
    num_classes=12,
    drop_path_rate=0.1
)
model.to(device)
