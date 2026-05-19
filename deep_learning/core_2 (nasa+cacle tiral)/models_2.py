import torch
import torch.nn as nn
import torch.nn.functional as F

class StochasticDepth(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep_prob = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask = torch.floor(rand)
        return x / keep_prob * mask


class ResBlock1D(nn.Module):
    def __init__(self, dim, p_sd):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, 3, padding=1)
        self.conv2 = nn.Conv1d(dim, dim, 3, padding=1)
        self.norm1 = nn.BatchNorm1d(dim)
        self.norm2 = nn.BatchNorm1d(dim)
        self.sd = StochasticDepth(p_sd)

    def forward(self, x):
        identity = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = self.sd(x)
        return F.gelu(x + identity)


class ResNet1D(nn.Module):
    def __init__(self, in_dim, p_sd):
        super().__init__()
        self.stem = nn.Conv1d(in_dim, 256, 1)
        self.blocks = nn.Sequential(*[ResBlock1D(256, p_sd) for _ in range(3)])

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, F, T]
        x = self.stem(x)
        return self.blocks(x)  # [B, 256, T]


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dropout):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout,
            batch_first=True, activation="gelu", norm_first=True
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x):
        return self.enc(x)


class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, seq_tokens, meta_token):
        attn_out, _ = self.attn(query=seq_tokens, key=meta_token, value=meta_token)
        seq_out = self.norm(seq_tokens + attn_out)
        return seq_out, meta_token


class DualReconstructionHead(nn.Module):
    def __init__(self, d_in, seq_dim, sum_dim):
        super().__init__()
        self.s_r = nn.Linear(d_in, seq_dim)
        self.d_r = nn.Linear(d_in, sum_dim)

    def forward(self, x):
        return self.s_r(x), self.d_r(x)


# 중요: 이 클래스는 들여쓰기 없이 맨 앞에 위치해야 합니다.
class MultiTaskRULModel(nn.Module):
    def __init__(self, sd, sm, c):
        super().__init__()
        self.c = c
        self.resnet = ResNet1D(sd, c.stochastic_depth_prob) if c.use_resnet else None
        self.proj_seq = nn.Linear(256 if c.use_resnet else sd, c.d_model)
        self.tr = TransformerEncoder(c.d_model, c.nhead, c.num_layers, c.dropout)
        self.pm = nn.Linear(sm, c.d_model)
        self.ca = CrossAttentionFusion(c.d_model, c.nhead)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.fus = nn.Sequential(nn.Linear(c.d_model * 2, 128), nn.GELU(), nn.Linear(128, 64))
        self.head = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        self.dual = DualReconstructionHead(c.d_model * 2, sd, sm) if c.dual_weight > 0 else None

    def forward(self, xs, xm):
        if self.resnet:
            seq_tokens = self.proj_seq(self.resnet(xs).transpose(1, 2))
        else:
            seq_tokens = self.proj_seq(xs)

        mt = self.pm(xm).unsqueeze(1)

        seq_tokens = torch.nan_to_num(seq_tokens, nan=0.0, posinf=0.0, neginf=0.0)
        mt = torch.nan_to_num(mt, nan=0.0, posinf=0.0, neginf=0.0)
        if hasattr(self.c, "token_clip"):
            seq_tokens = torch.clamp(seq_tokens, -self.c.token_clip, self.c.token_clip)

        seq_tokens = self.tr(seq_tokens)
        seq_out, sum_out = self.ca(seq_tokens, mt)

        seq_f = self.pool(seq_out.transpose(1, 2))
        sum_f = sum_out.squeeze(1)

        fused = self.fus(torch.cat([seq_f, sum_f], dim=1))
        head_out = self.head(fused)

        head_out = F.softplus(head_out)

        # [안정성 확보] Log 모드 Clamp 추가
        if getattr(self.c, "rul_mode", "minmax") == "minmax":
            head_out = torch.clamp(head_out, 0.0, 1.0)
        else:
            # log(2000) ~ 7.6 이므로 10.0으로 넉넉하게 제한하여 NaN 방지
            head_out = torch.clamp(head_out, 0.0, 10.0)

        if self.dual:
            rec_seq, rec_sum = self.dual(torch.cat([seq_f, sum_f], dim=1))
            return head_out, rec_seq, rec_sum, None

        return head_out, None, None, None