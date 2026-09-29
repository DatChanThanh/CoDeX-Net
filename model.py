import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import pathlib
import torchvision.models as models
import time
import os
from tqdm import tqdm
import timm
# Thiết bị sử dụng
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Đang sử dụng:", device)
print("Số GPU khả dụng:", torch.cuda.device_count())  # Kiểm tra số GPU

# Transform cho ResNet50
train_transform = transforms.Compose([
    #transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])
val_transform = transforms.Compose([
    #transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

# Đường dẫn dữ liệu
train_dir = pathlib.Path('D:\LAB_AI\dataset_new\\racomData12class_chia\\train')
val_dir = pathlib.Path('D:\LAB_AI\dataset_new\\racomData12class_chia\\test')
# Load dataset từ hai thư mục riêng
train_dataset = datasets.ImageFolder(root=str(train_dir), transform=train_transform)
val_dataset = datasets.ImageFolder(root=str(val_dir), transform=val_transform)

# DataLoader cho huấn luyện và test
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                          num_workers=4, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                        num_workers=4, pin_memory=True)

# Lấy thông tin lớp
class_names = train_dataset.classes
num_classes = len(class_names)
print(f"Number of classes: {num_classes}")
print(f"Class names: {class_names}")
print(f"Train samples: {len(train_dataset)}")       
print(f"Test samples: {len(val_dataset)}")
print(train_dataset.class_to_idx)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============== Small utils ===============
class DropPath(nn.Module):
    """Stochastic Depth (per sample)."""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.size(0),) + (1,)*(x.ndim-1)
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
    """Global Response Normalization (ConvNeXt-V2 style) for NCHW."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta  = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps
    def forward(self, x):
        gx = torch.sqrt((x**2).mean(dim=(2,3), keepdim=True) + self.eps)
        nx = x / (gx + self.eps)
        return x + self.gamma * nx + self.beta

# =============== CoordAttention (nhẹ, dài hạn theo H/W) ===============
class CoordAttention(nn.Module):
    """
    Coordinate Attention (Hou et al.): nén theo H và W rồi phục hồi mặt nạ chú ý.
    """
    def __init__(self, channels, reduction=32):
        super().__init__()
        m = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, m, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(m)
        self.act   = nn.SiLU(inplace=True)
        self.conv_h = nn.Conv2d(m, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(m, channels, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        x_h = F.adaptive_avg_pool2d(x, (H,1))             # (B,C,H,1)
        x_w = F.adaptive_avg_pool2d(x, (1,W))             # (B,C,1,W)
        x_w = x_w.permute(0,1,3,2)                        # (B,C,W,1)
        y = torch.cat([x_h, x_w], dim=2)                  # (B,C,H+W,1)
        y = self.act(self.bn1(self.conv1(y)))
        y_h, y_w = torch.split(y, [H, W], dim=2)
        a_h = torch.sigmoid(self.conv_h(y_h))             # (B,C,H,1)
        a_w = torch.sigmoid(self.conv_w(y_w).permute(0,1,3,2))  # (B,C,1,W)
        return x * a_h * a_w

# =============== 1) TinyGateV2: SwiGLU + residual gate + GRN ===============
class TinyGateV2(nn.Module):
    """
    Nâng cấp TinyGate: LN -> SwiGLU (Linear 2C) -> Linear C -> sigmoid,
    cộng GRN và residual scaling để gate ổn định & sắc nét hơn.
    """
    def __init__(self, channels, hidden=None, layerscale=1e-3):
        super().__init__()
        if hidden is None:
            hidden = max(8, channels // 3)
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.fc_in = nn.Linear(channels, hidden*2, bias=False)  # SwiGLU
        self.fc_out = nn.Linear(hidden, channels, bias=False)
        self.grn = GRN2d(channels)
        self.ls = LayerScale(channels, init_value=layerscale)

    def forward(self, x):
        B, C, _, _ = x.shape
        v = F.adaptive_avg_pool2d(x, 1).view(B, C)
        v = self.norm(v)
        a, b = self.fc_in(v).chunk(2, dim=-1)
        v = a * F.silu(b)                   # SwiGLU
        gate = torch.sigmoid(self.fc_out(v)).view(B, C, 1, 1)
        y = x * gate
        y = self.grn(y)
        return x + self.ls(y - x)          # residual gated update

# =============== 2) CrossLayerChannelFusionV2: token mixing + GEGLU + FiLM ===============
class CrossLayerChannelFusionV2(nn.Module):
    """
    - PreNorm (LN trên C ở N H W C)
    - Token mixer: DWConv 3x3 (local context) + GRN
    - Gated MLP: GEGLU (Linear 2H) -> Linear C
    - FiLM modulation: scale & bias từ GAP
    """
    def __init__(self, channels, expansion_factor=2):
        super().__init__()
        self.channels = channels
        hidden = channels * expansion_factor

        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.grn = GRN2d(channels)

        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.glu = nn.Linear(channels, hidden*2, bias=False)
        self.proj = nn.Linear(hidden, channels, bias=False)

        self.film = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, max(16, channels//2)), nn.SiLU(inplace=True),
            nn.Linear(max(16, channels//2), channels*2)  # (scale, bias)
        )

    def forward(self, x):
        # token mixer
        xm = self.grn(self.dw(x))

        # gated MLP per-pixel (operate on last dim)
        y = x.permute(0,2,3,1)                 # (B,H,W,C)
        y = self.norm(y)
        a, b = self.glu(y).chunk(2, dim=-1)    # GEGLU
        y = F.gelu(a) * b
        y = self.proj(y)                       # (B,H,W,C)
        y = y.permute(0,3,1,2).contiguous()    # (B,C,H,W)

        # FiLM modulation from GAP
        s, b = self.film(x).chunk(2, dim=-1)   # (B,C), (B,C)
        s = torch.sigmoid(s).view(x.size(0), self.channels, 1, 1)
        b = b.view(x.size(0), self.channels, 1, 1)

        return (xm + y) * s + b

# =============== 3) Gate_Spatial_Channel__UnitV2: SK-softmax + CoordAtt + DropPath ===============
class PAUG_PanAxisUnifiedGating(nn.Module):
    def __init__(self, channels, drop_path=0.0):
        super().__init__()
        self.C = channels
        # Channel branch (mới)
        self.channel_branch = CrossLayerChannelFusionV2(channels)

        # Spatial selective kernels
        self.dw1x3 = nn.Conv2d(channels, channels, kernel_size=(1,3), padding=(0,1),
                               groups=channels, bias=False)
        self.dw3x1 = nn.Conv2d(channels, channels, kernel_size=(3,1), padding=(1,0),
                               groups=channels, bias=False)
        self.dw3x3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1,
                               groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)

        # Softmax router + CoordAttention modulation
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(channels, max(4, channels//4)), nn.SiLU(inplace=True),
            nn.Linear(max(4, channels//4), 3)
        )
        self.coord = CoordAttention(channels, reduction=32)

        # Gate cuối: TinyGateV2 + residual tools
        self.gate = TinyGateV2(channels, hidden=max(8, channels//2))
        self.ls   = LayerScale(channels, init_value=1e-3)
        self.dp   = DropPath(drop_path)

    def forward(self, x):
        identity = x

        # Channel branch
        ch = self.channel_branch(x)

        # Spatial branch với SK-softmax + CoordAtt
        b1 = self.dw1x3(x); b2 = self.dw3x1(x); b3 = self.dw3x3(x)
        w = torch.softmax(self.router(x), dim=-1)  # (B,3)
        w1, w2, w3 = [w[:,i].view(-1,1,1,1) for i in range(3)]
        sp = w1*b1 + w2*b2 + w3*b3
        sp = self.act(self.bn(self.pw(sp)))
        sp = self.coord(sp)                        # spatial long-range guidance

        fused = ch + sp
        fused = self.gate(fused)                   # stronger gating
        out = identity + self.dp(self.ls(fused))   # residual + layerscale + SD
        return out

# =============== 4) FCSAV2: đa tỉ lệ với softmax router + scale-drop ===============
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==== per-scale unit (giữ nhẹ, ổn định) ====
class _DS(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.SiLU(inplace=True),
        )
        # Nếu bạn đã có TinyGateV2 & GRN2d, có thể thêm vào đây để mạnh hơn.
        self.gate = nn.Identity()
    def forward(self, x):
        return self.gate(self.block(x))

# ==== head dự đoán offset + attention ====
class OffsetAttnHead(nn.Module):
    """Từ đặc trưng truy vấn (full-res) dự đoán offset & trọng số cho K mẫu/điểm."""
    def __init__(self, channels, K=4, hidden_ratio=0.5):
        super().__init__()
        h = max(4, int(channels * hidden_ratio))
        self.conv = nn.Sequential(
            nn.Conv2d(channels, h, 3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(h, K*3, 1, bias=True)  # (K*2 offsets + K weights)
        )
        self.K = K

    def forward(self, q):                   # q: (B,C,Hf,Wf)
        B, C, H, W = q.shape
        out = self.conv(q)                  # (B, K*3, H, W)
        off, w = torch.split(out, [self.K*2, self.K], dim=1)
        off = off.view(B, self.K, 2, H, W)  # (B,K,2,H,W)
        w   = F.softmax(w.view(B, self.K, H, W), dim=1)  # (B,K,H,W)
        return off, w

# ==== tiện ích tạo lưới gốc (fine -> coarse, chuẩn hoá [-1,1]) ====
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
    coarse: (B,C,Hc,Wc) ; off: (B,K,2,Hf,Wf) ; w: (B,K,Hf,Wf)
    base_grid: (Hf,Wf,2) chuẩn hoá theo coarse; r_max: bán kính offset (pixel coarse)
    """
    B, C, Hc, Wc = coarse.shape
    B2, K, _, Hf, Wf = off.shape
    assert B == B2
    step_x = 2.0 / max(Wc - 1, 1)  # 1 pixel coarse theo hệ [-1,1]
    step_y = 2.0 / max(Hc - 1, 1)

    dx = torch.tanh(off[:, :, 0]) * (r_max * step_x)  # (B,K,Hf,Wf)
    dy = torch.tanh(off[:, :, 1]) * (r_max * step_y)
    delta = torch.stack((dx, dy), dim=-1)            # (B,K,Hf,Wf,2)

    base = base_grid.to(coarse.device, coarse.dtype).unsqueeze(0).unsqueeze(0)  # (1,1,Hf,Wf,2)
    grid = (base + delta).reshape(B*K, Hf, Wf, 2)

    src = coarse.unsqueeze(1).expand(B, K, C, Hc, Wc).reshape(B*K, C, Hc, Wc)
    samp = F.grid_sample(src, grid, mode='bilinear', padding_mode='border', align_corners=align_corners)
    samp = samp.view(B, K, C, Hf, Wf)               # (B,K,C,Hf,Wf)

    w = w.unsqueeze(2)                              # (B,K,1,Hf,Wf)
    return (w * samp).sum(dim=1)                    # (B,C,Hf,Wf)

# ==== Cross-Scale Deformable Mixer ====
class CSDM(nn.Module):
    """
    Cross-Scale Deformable Mixer:
      - Tạo 3 thang: full (s0), half (s1), quarter (s2)
      - Xử lý nhẹ mỗi thang (_DS)
      - Học offset+attention để lấy mẫu từ s1, s2 -> full qua grid_sample
      - Router softmax (3 nhánh: z0, y1, y2) rồi 1×1 + residual
    """
    def __init__(self, channels, K=4, r_max=2.0, head_hidden_ratio=0.5, router_hidden=None):
        super().__init__()
        C = channels
        self.proc0 = _DS(C)
        self.proc1 = _DS(C)
        self.proc2 = _DS(C)

        # query proj (tùy chọn, giữ C để nhẹ)
        self.qproj = nn.Conv2d(C, C, 1, bias=False)

        # hai head offset-attn riêng cho s1 và s2
        self.head1 = OffsetAttnHead(C, K=K, hidden_ratio=head_hidden_ratio)
        self.head2 = OffsetAttnHead(C, K=K, hidden_ratio=head_hidden_ratio)

        hrouter = router_hidden or max(16, C)
        self.router = nn.Sequential(
            nn.Linear(C*3, hrouter), nn.SiLU(inplace=True),
            nn.Linear(hrouter, 3)
        )
        self.final = nn.Conv2d(C, C, 1, bias=False)

        self.K = K
        self.r_max = float(r_max)

    def forward(self, x):
        B, C, H, W = x.shape

        # scales
        s0 = x
        s1 = F.adaptive_avg_pool2d(x, (max(1, H//2), max(1, W//2)))
        s2 = F.adaptive_avg_pool2d(x, (max(1, H//4), max(1, W//4)))

        # per-scale process
        z0 = self.proc0(s0)                 # (B,C,H,W)
        z1 = self.proc1(s1)                 # (B,C,H/2,W/2)
        z2 = self.proc2(s2)                 # (B,C,H/4,W/4)

        # query (full-res) dùng để dự đoán offset/attn
        q  = self.qproj(z0)                 # (B,C,H,W)

        # offsets & weights
        off1, w1 = self.head1(q)           # (B,K,2,H,W), (B,K,H,W)
        off2, w2 = self.head2(q)

        # base grids cho coarse->fine
        base1 = make_base_grid(H, W, z1.size(2), z1.size(3), q.device, q.dtype)
        base2 = make_base_grid(H, W, z2.size(2), z2.size(3), q.device, q.dtype)

        # deformable sampling từ z1,z2 -> full
        y1 = deform_sample(z1, off1, w1, base1, r_max=self.r_max)  # (B,C,H,W)
        y2 = deform_sample(z2, off2, w2, base2, r_max=self.r_max)

        # router softmax giữa 3 nhánh (z0, y1, y2)
        d0 = F.adaptive_avg_pool2d(z0,1).view(B,C)
        d1 = F.adaptive_avg_pool2d(y1,1).view(B,C)
        d2 = F.adaptive_avg_pool2d(y2,1).view(B,C)
        alpha = torch.softmax(self.router(torch.cat([d0,d1,d2], dim=-1)), dim=-1)  # (B,3)
        a0, a1, a2 = [alpha[:,i].view(B,1,1,1) for i in range(3)]

        out = a0*z0 + a1*y1 + a2*y2
        out = self.final(out)
        return x + out



class CoDeX_Net(nn.Module):

    def __init__(self, channels_per_stage=(8,8,16,16,32,32), num_classes=12, drop_path_rate=0.05):
        super().__init__()
        assert len(channels_per_stage) == 6
        self.chs = list(channels_per_stage)
        c0 = self.chs[0]

        # stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, c0, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c0),
            nn.SiLU(inplace=True),
            nn.Conv2d(c0, c0, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c0),
            nn.SiLU(inplace=True),
        )

        # schedule drop_path for 6 block
        total = len(self.chs)
        dprs = torch.linspace(0, drop_path_rate, steps=total).tolist()

        self.blocks = nn.ModuleList()
        for i, ch in enumerate(self.chs):
            in_ch = self.chs[i-1] if i>0 else c0
            proj = nn.Conv2d(in_ch, ch, 1, bias=False) if in_ch != ch else None
            module = PAUG_PanAxisUnifiedGating(ch, drop_path=dprs[i]) if (i % 2 == 0) else CSDM(channels=ch, K=4, r_max=2.0)
            self.blocks.append(nn.ModuleDict({"proj": proj, "module": module}))

        final_ch = self.chs[-1]
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_ch, max(128, final_ch*4)),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(max(128, final_ch*4), num_classes)
        )

    def forward(self, x):
        x = self.stem(x)
        for blk in self.blocks:
            if blk["proj"] is not None: x = blk["proj"](x)
            x = blk["module"](x)
        x = self.global_pool(x).view(x.size(0), -1)
        return self.classifier(x)


channels = (8, 8, 16, 16, 32, 32)
model = CoDeX_Net(channels_per_stage=channels, num_classes=12, drop_path_rate=0.1)
    
model.to(device)
import os
import torch
from torchvision import transforms
from PIL import Image
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total number of parameters: {total_params}")
print(f"Number of trainable parameters: {trainable_params}")
# Hàm train và validate
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    with tqdm(dataloader, desc="Train", colour="cyan", ncols=100) as pbar:
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix(loss=running_loss/total, accuracy=f"{correct / total:.4f}")
            pbar.update(1)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with tqdm(dataloader, desc="Test", colour="cyan", ncols=100) as pbar:
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix(loss=running_loss / total, accuracy=f"{correct / total:.4f}")
                pbar.update(1)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# Train model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3, min_lr=1e-6)

start_epoch = 0
num_epochs = 60
best_acc = 0.0
best_epoch = start_epoch

resume_train = 'F:\THANHDAT\\final_model_epoch_53_acc_0.9020.pt'
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

if os.path.exists(resume_train):
    checkpoint = torch.load(resume_train, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint.get('val_acc', 0.0)
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    train_accuracies = checkpoint.get('train_accuracies', [])
    val_accuracies = checkpoint.get('val_accuracies', [])
    print(f"Resumed training from epoch {start_epoch} with val_acc: {best_acc:.4f}")
else:
    print(f"Checkpoint {resume_train} not found. Starting from scratch.")

# Train và validate
for epoch in range(start_epoch, num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    current_lr = optimizer.param_groups[0]['lr']

    start_time = time.time()
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    elapsed = time.time() - start_time

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}")
    print(f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")
    print(f"time/epoch: {elapsed:.1f}s, current LR: {current_lr}")

    scheduler.step(val_loss)

    if val_acc > best_acc:
        best_acc = val_acc
        best_model_path = f"best_model_epoch_{epoch+1}_acc_{best_acc:.4f}.pt"
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Saved best model at epoch {epoch+1} with val_acc: {best_acc:.4f}")

    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_acc': val_acc,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'train_loss': train_loss,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }
    final_model_path = f"final_model_epoch_{epoch+1}_acc_{val_acc:.4f}.pt"
    torch.save(checkpoint, final_model_path)
    print(f"💾 Saved final checkpoint at epoch {epoch+1} with val_acc: {val_acc:.4f}")
