import math
import torch
from torch import nn
from torch.nn import functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module): 
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb, freeze=False), 
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.timembedding(t)
        return emb
    
class ConditionalEmbedding(nn.Module): 
    def __init__(self, d_model):
        super().__init__()
        self.conv_embedding = nn.Sequential(
            nn.Conv2d(3, d_model // 2, kernel_size=3, stride=2, padding=1),  
            nn.GroupNorm(8, d_model // 2),
            Swish(),
            nn.Conv2d(d_model // 2, d_model, kernel_size=3, stride=2, padding=1),  
            nn.GroupNorm(8, d_model),
            Swish()
        )

    def forward(self, labels):
        emb = self.conv_embedding(labels)  
        return emb 

class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.c1 = nn.Conv3d(in_ch, in_ch, kernel_size=3, stride=2, padding=1)  # 3x3x3 
        self.c2 = nn.Conv3d(in_ch, in_ch, kernel_size=5, stride=2, padding=2)  # 5x5x5

    def forward(self, x, temb, cemb1, cemb2, cemb3, face1, face2, face3):
        x = self.c1(x) + self.c2(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.t = nn.ConvTranspose3d(in_ch, in_ch, kernel_size=5, stride=2, padding=2, output_padding=1)  
        self.c = nn.Conv3d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)  

    def forward(self, x, temb, cemb1, cemb2, cemb3, face1, face2, face3):
        x = self.t(x)
        x = self.c(x)
        return x

class AttnBlock(nn.Module):  
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(8, in_ch)   
        self.proj_q = nn.Conv3d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv3d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv3d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv3d(in_ch, in_ch, 1, stride=1, padding=0)

    def forward(self, x): 
        B, C, M, H, W = x.shape # (batch, channels, depth, height, width)
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 4, 1).view(B, M * H * W, C)
        k = k.view(B, C, M * H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, M * H * W, M * H * W]
        w = F.softmax(w, dim=-1) 

        v = v.permute(0, 2, 3, 4, 1).view(B, M * H * W, C)
        h = torch.bmm(w, v) 
        assert list(h.shape) == [B, M * H * W, C]
        h = h.view(B, M, H, W, C).permute(0, 4, 1, 2, 3)
        h = self.proj(h)

        return x + h   

class AdditionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.decay_rate_d = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))  # D direction
        self.decay_rate_h = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))  # H direction
        self.decay_rate_w = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))  # W direction

        """
        h:          (B, c, D, H, W)  
        cond_emb:   (B, c, H2D, W2D) 
        """
    def forward(self, h, cond_emb, face_id_map):
        B, c3d, D, H, W = h.shape
        B2, c2d, H2, W2 = cond_emb.shape
        assert B == B2, "batch size not match"
        assert c3d == c2d, "channel not match"

        if (H2 != H) or (W2 != W):
            cond_emb = F.interpolate(cond_emb, size=(H, W), mode='bicubic')

        # calculate decay weights
        decay_weight_d_f = torch.exp(-self.decay_rate_d * torch.arange(D, dtype=torch.float32, device=h.device)) # dacay in Z axis
        decay_weight_d_r = torch.exp(-self.decay_rate_d * torch.arange(D-1, -1, -1, dtype=torch.float32, device=h.device))  # dacay in z direction
        decay_weight_h_f = torch.exp(-self.decay_rate_h * torch.arange(H, dtype=torch.float32, device=h.device))  # dacay in Y direction
        decay_weight_h_r = torch.exp(-self.decay_rate_h * torch.arange(H-1, -1, -1, dtype=torch.float32, device=h.device))  # dacay in Y direction
        decay_weight_w_f = torch.exp(-self.decay_rate_w * torch.arange(W, dtype=torch.float32, device=h.device)) # dacay in X direction
        decay_weight_w_r = torch.exp(-self.decay_rate_w * torch.arange(W-1, -1, -1, dtype=torch.float32, device=h.device))  # dacay in X direction
        
        decay_weight_d_f = decay_weight_d_f.view(1, D, 1, 1)  # (1, D, 1, 1)
        decay_weight_w_f = decay_weight_w_f.view(1, 1, 1, W)  # (1, 1, 1, W)
        decay_weight_h_f = decay_weight_h_f.view(1, 1, H, 1)  # (1, 1, H, 1)

        decay_weight_d_r = decay_weight_d_r.view(1, D, 1, 1)  # (1, D, 1, 1)
        decay_weight_w_r = decay_weight_w_r.view(1, 1, 1, W)  # (1, 1, 1, W)
        decay_weight_h_r = decay_weight_h_r.view(1, 1, H, 1)  # (1, 1, H, 1)

        decay_weight_d_f = decay_weight_d_f.expand(h.shape[1], -1, h.shape[3], h.shape[4])
        decay_weight_w_f = decay_weight_w_f.expand(h.shape[1], h.shape[2], h.shape[3], -1)
        decay_weight_h_f = decay_weight_h_f.expand(h.shape[1], h.shape[2], -1, h.shape[4])

        decay_weight_d_r = decay_weight_d_r.expand(h.shape[1], -1, h.shape[3], h.shape[4])
        decay_weight_w_r = decay_weight_w_r.expand(h.shape[1], h.shape[2], h.shape[3], -1)
        decay_weight_h_r = decay_weight_h_r.expand(h.shape[1], h.shape[2], -1, h.shape[4])

        for b in range(B):
            face_val = face_id_map[b].item() 
            emb_2d = cond_emb[b] # (c, H, W)
            emb_2d_d = emb_2d[ :, None, :, :].expand(-1, h.shape[2], -1, -1)  
            emb_2d_w = emb_2d[ :, :, :, None].expand(-1, -1, -1, h.shape[2])
            emb_2d_h = emb_2d[ :, :, None, :].expand(-1, -1, h.shape[2], -1)

            if face_val == 1:  # front (-Z) => d=0
                weight = decay_weight_d_f 
                h[b] += emb_2d_d * weight

            elif face_val == 2:  # back  (+Z) => d=D-1
                weight = decay_weight_d_r  
                h[b] += emb_2d_d * weight

            elif face_val == 3:  # left (-X) => w=0
                weight = decay_weight_w_f  
                h[b] += emb_2d_w * weight

            elif face_val == 4:  # right (+X) => w=W-1
                weight = decay_weight_w_r  
                h[b] += emb_2d_w * weight

            elif face_val == 5:  # top (+Y) => h=0
                weight = decay_weight_h_f 
                h[b] += emb_2d_h * weight

            elif face_val == 6:  # bottom (-Y) => h=H-1
                weight = decay_weight_h_r
                h[b] += emb_2d_h * weight
        
        return h

class ResBlock(nn.Module):  
    def __init__(self, in_ch, out_ch, tdim, dropout, ch, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            Swish(),
            nn.Conv3d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )

        self.cond_proj = nn.Sequential(
            nn.Conv2d(ch, out_ch, kernel_size=3, stride=1, padding=1),  
            nn.GroupNorm(8, out_ch),
            Swish(),
        )
        self.addition = AdditionBlock()

        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv3d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()

    def forward(self, x, temb, cemb1, cemb2, cemb3, face1, face2, face3):
        h = self.block1(x)
        h += self.temb_proj(temb)[:, :, None, None, None] 

        cond_emb1 = self.cond_proj(cemb1)
        cond_emb2 = self.cond_proj(cemb2)
        cond_emb3 = self.cond_proj(cemb3)

        h = self.addition(h, cond_emb1, face1)
        h = self.addition(h, cond_emb2, face2)
        h = self.addition(h, cond_emb3, face3)

        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h 

class UNet(nn.Module): 
    def __init__(self, T, ch, ch_mult, num_res_blocks, dropout):
        super().__init__()
        tdim = ch * 4
        self.time_embedding = TimeEmbedding(T, ch, tdim)
        self.cond_embedding = ConditionalEmbedding(ch)

        self.head = nn.Conv3d(1, ch, kernel_size=3, stride=1, padding=1) 
        self.downblocks = nn.ModuleList()
        chs = [ch]  
        now_ch = ch

        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(num_res_blocks):
                self.downblocks.append(ResBlock(in_ch=now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, ch=ch))
                now_ch = out_ch
                chs.append(now_ch)
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(now_ch))
                chs.append(now_ch)

        self.middleblocks = nn.ModuleList([
            ResBlock(now_ch, now_ch, tdim, dropout, ch=ch, attn=True),
            ResBlock(now_ch, now_ch, tdim, dropout, ch=ch, attn=True),
        ])

        self.upblocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out_ch = ch * mult
            for _ in range(num_res_blocks + 1):
                self.upblocks.append(ResBlock(in_ch=chs.pop() + now_ch, out_ch=out_ch, tdim=tdim, dropout=dropout, ch=ch))
                now_ch = out_ch
            if i != 0:
                self.upblocks.append(UpSample(now_ch))
        assert len(chs) == 0

        self.tail = nn.Sequential(
            nn.GroupNorm(8, now_ch),
            Swish(),
            nn.Conv3d(now_ch, 1, 3, stride=1, padding=1)
        )

    def forward(self, x, t, labels):
        temb = self.time_embedding(t)
        cemb1 = self.cond_embedding(labels[:,:3,:,:]) # the first face + features (porosity, specific surface area)
        cemb2 = self.cond_embedding(labels[:,4:7,:,:])  # the second face + features 
        cemb3 = self.cond_embedding(labels[:,8:11,:,:]) # the third face + features 

        B, _, _, _ = cemb1.shape
        face1 = torch.unique(labels[:,3:4,:,:].view(B,-1), dim=1).squeeze(1)
        face2 = torch.unique(labels[:,7:8,:,:].view(B,-1), dim=1).squeeze(1)
        face3 = torch.unique(labels[:,11:12,:,:].view(B,-1), dim=1).squeeze(1)

        # Downsampling 
        h = self.head(x)
        hs = [h]
        for layer in self.downblocks:
            h = layer(h, temb, cemb1, cemb2, cemb3, face1, face2, face3)
            hs.append(h)

        # Middle blocks
        for layer in self.middleblocks:
            h = layer(h, temb, cemb1, cemb2, cemb3, face1, face2, face3)

        # Upsampling 
        for layer in self.upblocks:
            if isinstance(layer, ResBlock):
                prev_feature = hs.pop()
                if h.shape[2:] != prev_feature.shape[2:]:
                    h = F.interpolate(h, size=prev_feature.shape[2:], mode='trilinear', align_corners=False)
                h = torch.cat([h, prev_feature], dim=1)
            h = layer(h, temb, cemb1, cemb2, cemb3, face1, face2, face3)
        h = self.tail(h)

        assert len(hs) == 0
        return h

