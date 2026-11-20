"""Neural blocks for the three pipelines.

    speech : [frozen WavLM layers] -> LayerWeightedProjection -> BiGRU x2 -> AttentivePooling   (temporal block)
    text   : [frozen DistilRoBERTa layers] -> LayerWeightedProjection -> BiGRU -> AttentivePooling (contextual block)
    fusion : GatedFusion (GMU) | ConcatFusion | CrossAttentionFusion
    head   : MLP classifier
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .config import NUM_CLASSES


# ---------------------------------------------------------------- feature-level blocks
class LayerWeightedProjection(nn.Module):
    """Learnable softmax-weighted sum over the frozen encoder's layers (SUPERB recipe),
    followed by LayerNorm + linear projection. Lets the model pick the depth at
    which emotion (paralinguistic for speech, semantic for text) is encoded."""

    def __init__(self, num_layers: int, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.GELU(), nn.Dropout(dropout))

    def layer_weights(self) -> torch.Tensor:
        return self.layer_logits.softmax(0)

    def forward(self, x):  # x: (B, T, L, D)
        x = (x * self.layer_weights()[None, None, :, None]).sum(2)
        return self.proj(self.norm(x))


class ConvFrontend(nn.Module):
    """For the hand-crafted log-Mel baseline: 2 x (Conv1d k=5 + BN + GELU)."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 128, 5, padding=2), nn.BatchNorm1d(128), nn.GELU(),
            nn.Conv1d(128, out_dim, 5, padding=2), nn.BatchNorm1d(out_dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):  # (B, T, 1, F)
        return self.net(x[:, :, 0].transpose(1, 2)).transpose(1, 2)


def time_mask(x: torch.Tensor, lengths: torch.Tensor, n_masks: int = 2, max_frac: float = 0.1):
    """SpecAugment-style time masking applied to frame embeddings (training only)."""
    B, T = x.shape[:2]
    L = lengths.to(x.device).float()[:, None]                                     # (B,1)
    w = (torch.rand(B, n_masks, device=x.device) * (L * max_frac + 1)).floor()   # mask widths
    s = (torch.rand(B, n_masks, device=x.device) * (L - w).clamp(min=1)).floor()  # mask starts
    t = torch.arange(T, device=x.device)[None, None, :]
    masked = ((t >= s[..., None]) & (t < (s + w)[..., None])).any(1)             # (B,T)
    return x.masked_fill(masked[..., None], 0.0)


# ---------------------------------------------------------------- sequence blocks
class BiGRU(nn.Module):
    def __init__(self, dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.rnn = nn.GRU(dim, dim // 2, num_layers=num_layers, batch_first=True, bidirectional=True,
                          dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        return out


class AttentivePooling(nn.Module):
    """Additive attention over time: which frames/tokens carry the emotion."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(dim, dim // 2), nn.Tanh(), nn.Linear(dim // 2, 1))

    def forward(self, x, mask):  # x (B,T,D), mask (B,T) bool, True = valid
        s = self.score(x).squeeze(-1).masked_fill(~mask, float("-inf"))
        a = s.softmax(-1)
        return torch.bmm(a.unsqueeze(1), x).squeeze(1), a


class SequenceEncoder(nn.Module):
    """Feature projection -> BiGRU -> attentive pooling. Used as the speech
    *temporal-modelling* block and the text *contextual-modelling* block."""

    def __init__(self, num_layers_in: int, in_dim: int, dim: int = 256, rnn_layers: int = 2,
                 dropout: float = 0.2, frontend: str = "layerweight", augment: bool = False):
        super().__init__()
        if frontend == "conv":
            self.frontend = ConvFrontend(in_dim, dim, dropout)
        else:
            self.frontend = LayerWeightedProjection(num_layers_in, in_dim, dim, dropout)
        self.rnn = BiGRU(dim, rnn_layers, dropout)
        self.pool = AttentivePooling(dim)
        self.augment = augment
        self.out_dim = dim

    def forward(self, x, lengths, mask):
        h = self.frontend(x)
        if self.training and self.augment:
            h = time_mask(h, lengths)
        seq = self.rnn(h, lengths)
        pooled, attn = self.pool(seq, mask)
        return seq, pooled, attn


class StatsPoolEncoder(nn.Module):
    """Head-capacity study (no recurrence): softmax layer-weighted sum of the frozen layers ->
    LayerNorm -> dropout -> attentive statistics pooling (attention-weighted mean and std over frames,
    as in ECAPA-TDNN) -> linear projection to `dim`. The only temporal operation is the attention
    weighting; frame order is ignored."""

    def __init__(self, num_layers_in: int, in_dim: int, dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(num_layers_in))
        self.norm = nn.LayerNorm(in_dim)
        self.drop = nn.Dropout(dropout)
        self.score = nn.Linear(in_dim, 1)
        self.proj = nn.Linear(2 * in_dim, dim)
        self.out_dim = dim

    def layer_weights(self) -> torch.Tensor:
        return self.layer_logits.softmax(0)

    def forward(self, x, lengths, mask):  # x (B,T,L,D)
        h = self.drop(self.norm((x * self.layer_weights()[None, None, :, None]).sum(2)))
        a = self.score(h).squeeze(-1).masked_fill(~mask, float("-inf")).softmax(-1)      # (B,T)
        mean = torch.bmm(a.unsqueeze(1), h).squeeze(1)
        var = torch.bmm(a.unsqueeze(1), h * h).squeeze(1) - mean * mean
        pooled = self.proj(torch.cat([mean, var.clamp(min=1e-6).sqrt()], -1))
        return h, pooled, a


# ---------------------------------------------------------------- fusion blocks
class ConcatFusion(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, s, t, **_):
        return self.net(torch.cat([s, t], -1)), None


class GatedFusion(nn.Module):
    """Gated Multimodal Unit (Arevalo et al., 2017).
        h_s = tanh(W_s s), h_t = tanh(W_t t), z = sigmoid(W_z [s; t])
        h   = z * h_s + (1 - z) * h_t
    The per-dimension gate z is input-dependent, so the network can rely on
    speech when text is uninformative (and vice versa) - and z is inspectable."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.hs, self.ht = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.gate = nn.Linear(2 * dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, s, t, **_):
        z = torch.sigmoid(self.gate(torch.cat([s, t], -1)))
        h = z * torch.tanh(self.hs(s)) + (1 - z) * torch.tanh(self.ht(t))
        return self.drop(h), z


class CrossAttentionFusion(nn.Module):
    """Bidirectional cross-modal attention between frame and token sequences,
    then attentive pooling of each attended stream and a joint projection."""

    def __init__(self, dim: int, dropout: float, heads: int = 4):
        super().__init__()
        self.s2t = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.t2s = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ns, self.nt = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ps, self.pt = AttentivePooling(dim), AttentivePooling(dim)
        self.out = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, s, t, s_seq=None, t_seq=None, s_mask=None, t_mask=None):
        s_att, _ = self.s2t(s_seq, t_seq, t_seq, key_padding_mask=~t_mask)
        t_att, _ = self.t2s(t_seq, s_seq, s_seq, key_padding_mask=~s_mask)
        s_new, _ = self.ps(self.ns(s_seq + s_att), s_mask)
        t_new, _ = self.pt(self.nt(t_seq + t_att), t_mask)
        return self.out(torch.cat([s_new, t_new], -1)), None


FUSIONS = {"gated": GatedFusion, "concat": ConcatFusion, "xattn": CrossAttentionFusion}


class Classifier(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3, n_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim // 2, n_classes))

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------- full models
def _speech_encoder(meta, cfg):
    if cfg.get("head", "bigru") == "attnpool":
        return StatsPoolEncoder(meta["speech_layers"], meta["speech_dim"], cfg["dim"], cfg["dropout"])
    conv = cfg["speech_features"] == "logmel"
    return SequenceEncoder(meta["speech_layers"], meta["speech_dim"], cfg["dim"], rnn_layers=2,
                           dropout=cfg["dropout"], frontend="conv" if conv else "layerweight", augment=True)


def _text_encoder(meta, cfg):
    return SequenceEncoder(meta["text_layers"], meta["text_dim"], cfg["dim"], rnn_layers=1,
                           dropout=cfg["dropout"], frontend="layerweight", augment=False)


class SpeechModel(nn.Module):
    def __init__(self, meta, cfg):
        super().__init__()
        self.encoder = _speech_encoder(meta, cfg)
        self.classifier = Classifier(cfg["dim"])

    def forward(self, b):
        _, pooled, attn = self.encoder(b["speech"], b["speech_len"], b["speech_mask"])
        return self.classifier(pooled), {"temporal": pooled}, {"speech_attn": attn}


class TextModel(nn.Module):
    def __init__(self, meta, cfg):
        super().__init__()
        self.encoder = _text_encoder(meta, cfg)
        self.classifier = Classifier(cfg["dim"])

    def forward(self, b):
        _, pooled, attn = self.encoder(b["text"], b["text_len"], b["text_mask"])
        return self.classifier(pooled), {"contextual": pooled}, {"text_attn": attn}


class FusionModel(nn.Module):
    def __init__(self, meta, cfg):
        super().__init__()
        self.speech = _speech_encoder(meta, cfg)
        self.text = _text_encoder(meta, cfg)
        self.fusion = FUSIONS[cfg["fusion"]](cfg["dim"], cfg["dropout"])
        self.classifier = Classifier(cfg["dim"])
        self.modality_dropout = cfg.get("modality_dropout", 0.0)

    def forward(self, b):
        s_seq, s, s_attn = self.speech(b["speech"], b["speech_len"], b["speech_mask"])
        t_seq, t, t_attn = self.text(b["text"], b["text_len"], b["text_mask"])
        if self.training and self.modality_dropout > 0:
            # randomly silence one modality per sample so neither branch is ignored
            B = s.size(0)
            r = torch.rand(B, device=s.device)
            keep_s = (r >= self.modality_dropout / 2).float()[:, None]
            keep_t = ((r < self.modality_dropout / 2) | (r >= self.modality_dropout)).float()[:, None]
            s, t = s * keep_s, t * keep_t
            s_seq, t_seq = s_seq * keep_s[:, :, None], t_seq * keep_t[:, :, None]
        fused, gate = self.fusion(s, t, s_seq=s_seq, t_seq=t_seq,
                                  s_mask=b["speech_mask"], t_mask=b["text_mask"])
        extras = {"speech_attn": s_attn, "text_attn": t_attn}
        if gate is not None:
            extras["gate"] = gate
        return self.classifier(fused), {"temporal": s, "contextual": t, "fusion": fused}, extras


MODELS = {"speech": SpeechModel, "text": TextModel, "fusion": FusionModel}
