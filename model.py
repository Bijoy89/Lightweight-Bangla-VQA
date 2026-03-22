import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3
IMG_CHANNELS = 1280


class QuestionGuidedAttention(nn.Module):
    def __init__(self, img_channels, q_dim, hidden_dim):
        super().__init__()
        self.img_proj = nn.Linear(img_channels, hidden_dim)
        self.q_proj   = nn.Linear(q_dim, hidden_dim)
        self.attn     = nn.Linear(hidden_dim, 1)

    def forward(self, feat_map, q_vec):
        B, C, H, W = feat_map.shape
        locs    = feat_map.view(B, C, -1).permute(0, 2, 1)
        img_h   = self.img_proj(locs)
        q_h     = self.q_proj(q_vec).unsqueeze(1)
        scores  = self.attn(torch.tanh(img_h + q_h))
        weights = F.softmax(scores, dim=1)
        ctx     = (locs * weights).sum(dim=1)
        return ctx, weights.squeeze(-1)


class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.wv = nn.Linear(dim, dim)
        self.wq = nn.Linear(dim, dim)
        self.wp = nn.Linear(dim, dim)

    def forward(self, v, q):
        g = torch.sigmoid(self.wv(v) + self.wq(q))
        return self.wp(g * v + (1 - g) * q)


class VQAModel(nn.Module):
    """
    EfficientNet-B0 → QuestionGuidedAttention → Concat/Gated Fusion
    → BiLSTM → Autoregressive LSTM Decoder
    Supports both concat and gated fusion types.
    Best config from training: fusion=concat, hidden=512, emb=300
    """
    def __init__(
        self,
        q_vocab_size, a_vocab_size,
        hidden_dim=512,        # updated to 512 (best from Stage 1)
        fusion_type='concat',  #  updated to concat (best from Stage 1)
        emb_dim=300,
        img_channels=IMG_CHANNELS,
        dropout=0.3,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.hidden_dim  = hidden_dim

        # Question encoder
        self.q_embed = nn.Embedding(q_vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.q_lstm  = nn.LSTM(emb_dim, hidden_dim, num_layers=1,
                               batch_first=True, bidirectional=True)
        self.q_proj  = nn.Linear(hidden_dim * 2, hidden_dim)

        # Spatial attention
        self.attention = QuestionGuidedAttention(img_channels, hidden_dim, hidden_dim)

        # Image projection
        self.img_proj = nn.Sequential(
            nn.Linear(img_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Fusion
        if fusion_type == 'gated':
            self.fusion    = GatedFusion(hidden_dim)
            self.fused_dim = hidden_dim
        else:  # concat
            self.fusion    = None
            self.fused_dim = hidden_dim * 2

        # Decoder
        self.a_embed  = nn.Embedding(a_vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.dec_lstm = nn.LSTM(emb_dim + self.fused_dim, hidden_dim,
                                num_layers=1, batch_first=True)
        self.dec_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, a_vocab_size)

    def encode(self, feat_map, q_ids):
        q_emb    = self.q_embed(q_ids)
        q_out, _ = self.q_lstm(q_emb)
        q_vec    = self.q_proj(q_out[:, -1, :])
        ctx, attn_weights = self.attention(feat_map, q_vec)
        v        = self.img_proj(ctx)
        if self.fusion_type == 'gated':
            fused = self.fusion(v, q_vec)
        else:
            fused = torch.cat([v, q_vec], dim=-1)
        return fused, q_vec, attn_weights

    @torch.no_grad()
    def beam_search(self, feat_map, q_ids, beam_size=3, max_len=10):
        self.eval()
        fused, _, _ = self.encode(feat_map, q_ids)
        beams       = [(0.0, [SOS_IDX], None)]
        done        = []

        for _ in range(max_len):
            candidates = []
            for log_p, toks, hc in beams:
                last        = torch.tensor([toks[-1]], device=DEVICE)
                emb         = self.a_embed(last)
                inp         = torch.cat([emb, fused], dim=-1).unsqueeze(1)
                out, new_hc = self.dec_lstm(inp, hc)
                logit       = self.out_proj(out.squeeze(1))
                log_probs   = F.log_softmax(logit, dim=-1)[0]
                topk_lp, topk_ids = log_probs.topk(beam_size)
                for lp, tid in zip(topk_lp.tolist(), topk_ids.tolist()):
                    seq = toks + [tid]
                    if tid == EOS_IDX:
                        done.append((log_p + lp, seq))
                    else:
                        candidates.append((log_p + lp, seq, new_hc))
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_size]
            if not beams:
                break

        if done:
            done.sort(key=lambda x: x[0] / len(x[1]), reverse=True)
            best = done[0][1]
        else:
            best = beams[0][1]

        return best[1:]  # skip SOS