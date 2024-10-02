import torch
import torch.nn as nn
import torch.nn.functional as F

import math

# freq vers.
class Transformer(nn.Module):
    def __init__(self, n_feat, n_embed, n_layer_enc, n_layer_dec, n_head, attn_pdrop=.1, resid_pdrop=.1,
                 mlp_hid_times=4, dropout=0.0, max_len=2048):
        super(Transformer, self).__init__()
        # sine, mujoco, energy, ...
        self.embed = Embedding_conv(in_ch=n_feat, out_ch=n_embed, kernel_size=3, dropout=dropout)
        self.embed_lf = Embedding_conv(in_ch=n_feat, out_ch=n_embed, kernel_size=3, dropout=dropout)
        self.embed_hf = Embedding_conv(in_ch=n_feat, out_ch=n_embed, kernel_size=3, dropout=dropout)
        self.out = Embedding_conv(in_ch=n_embed, out_ch=n_feat, kernel_size=3, dropout=dropout, final=True)


        self.pos_enc = LearnablePositionalEncoding(d_model=n_embed, dropout=dropout, max_len=max_len)
        self.pos_dec = LearnablePositionalEncoding(d_model=n_embed, dropout=dropout, max_len=max_len)

        freq_len = math.ceil((max_len + 1) / 2)
        self.pos_lf = LearnablePositionalEncoding(d_model=n_embed, dropout=dropout, max_len=freq_len)
        self.pos_hf = LearnablePositionalEncoding(d_model=n_embed, dropout=dropout, max_len=freq_len)

        self.encoder = Encoder(n_layer=n_layer_enc, n_embed=n_embed, n_head=n_head, attn_pdrop=attn_pdrop,
                               resid_pdrop=resid_pdrop, mlp_hid_times=mlp_hid_times, block_activate='GELU')
        self.decoder = Decoder(n_feat=n_feat, n_layer=n_layer_dec, n_embed=n_embed, n_head=n_head,
                               condition_dim=n_embed, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop,
                               mlp_hid_times=mlp_hid_times, block_activate="GELU")


    def forward(self, x, t, condition=None, padding_masks=None):
        # x: [batch, time, channel], condition: [low frequency, high frequency]
        x = self.embed(x)
        x_enc = self.pos_enc(x)
        enc_out = self.encoder(x_enc, t, padding_masks)

        # freq parts
        lf = self.embed_lf(condition[0])
        hf = self.embed_hf(condition[1])

        lf = self.pos_lf(lf)
        hf = self.pos_hf(hf)

        x_dec = self.pos_dec(x)
        output, mean = self.decoder(x_dec, t, enc_out, [lf, hf], padding_masks=padding_masks)

        res = self.out(output)
        res_m = torch.mean(res, dim=1, keepdim=True)
        return res, res_m

class Embedding_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dropout=0., final=False):
        super(Embedding_conv, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Dropout(dropout)
        )
        if final:
            self.final_layer = nn.Conv1d(in_channels=out_ch, out_channels=out_ch, kernel_size=1)
        self.final = final

    def forward(self, x):
        x = self.layers(x.transpose(1, 2))
        if self.final:
            x = self.final_layer(x)
        return x.transpose(1, 2)

class EncoderBlock(nn.Module):
    def __init__(self, n_embed=1024, n_head=16, mlp_hid_times=4, attn_pdrop=.1, resid_pdrop=.1, activate='GELU'):
        super(EncoderBlock, self).__init__()
        self.ln1 = AdaLayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.attn = FullAttention(
            n_embed=n_embed,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop
        )
        self.mlp = nn.Sequential(
            nn.Linear(n_embed, mlp_hid_times * n_embed),
            nn.GELU(),
            nn.Linear(mlp_hid_times*n_embed, n_embed),
            nn.Dropout(resid_pdrop)
        )

    def forward(self, x, timestep, mask=None, label_emb=None):
        a, att = self.attn(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        x = x + self.mlp(self.ln2(x))   # only one really use encoder_output
        return x, att

class Encoder(nn.Module):
    def __init__(self, n_layer=14, n_embed=1024, n_head=16, attn_pdrop=.0, resid_pdrop=.0, mlp_hid_times=4, block_activate='GELU'):
        super(Encoder, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(n_layer):
            self.blocks.append(
                EncoderBlock(n_embed=n_embed,
                             n_head=n_head,
                             attn_pdrop=attn_pdrop,
                             resid_pdrop=resid_pdrop,
                             mlp_hid_times=mlp_hid_times,
                             activate=block_activate)
            )

    def forward(self, x, t, padding_mask=None, label_emb=None):
        for block in self.blocks:
            x, _ = block(x, t, padding_mask, label_emb)
        return x

# freq vers.
class DecoderBlock(nn.Module):
    def __init__(self, n_feat, n_embed=1024, n_head=16, condition_dim=512, mlp_hid_times=4, attn_pdrop=.1, resid_pdrop=.1, activate='GELU'):
        super(DecoderBlock, self).__init__()
        self.ln1 = AdaLayerNorm(n_embed)
        self.ln1_1 = AdaLayerNorm(n_embed)
        self.ln1_lf = AdaLayerNorm(n_embed)
        self.ln1_hf = AdaLayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ln2_fq = nn.LayerNorm(n_embed*2)

        self.attn1 = FullAttention(
            n_embed=n_embed,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop
        )

        self.attn2 = CrossAttention(
            n_embed=n_embed,
            condition_embed=condition_dim,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop
        )

        self.attn3 = CrossAttention(
            n_embed=n_embed,
            condition_embed=condition_dim,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop
        )

        self.attn4 = CrossAttention(
            n_embed=n_embed,
            condition_embed=condition_dim,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop
        )

        self.mlp = nn.Sequential(
            nn.Linear(n_embed, mlp_hid_times * n_embed),
            nn.GELU(),
            nn.Linear(mlp_hid_times * n_embed, n_embed),
            nn.Dropout(resid_pdrop)
        )

        self.mlp_fq = nn.Sequential(
            nn.Linear(n_embed*2, mlp_hid_times * n_embed),
            nn.GELU(),
            nn.Linear(mlp_hid_times * n_embed, n_embed),
            nn.Dropout(resid_pdrop)
        )

        self.linear = nn.Linear(n_embed, n_feat)

    def forward(self, x, enc_out, freq_out, timestep, mask=None, label_emb=None):
        a, att = self.attn1(self.ln1(x, timestep, label_emb), mask=mask)
        x = x + a
        a, att = self.attn2(self.ln1_1(x, timestep), enc_out, mask=mask)
        x = x + a
        # frequency parts
        lf, lf_att = self.attn3(self.ln1_lf(x, timestep), freq_out[0], mask=mask)
        hf, hf_att = self.attn4(self.ln1_hf(x, timestep), freq_out[1], mask=mask)

        fq = self.mlp_fq(self.ln2_fq(torch.cat([lf, hf], dim=2)))

        x = x + self.mlp(self.ln2(x)) + fq
        m = torch.mean(x, dim=1, keepdim=True)
        return x - m, self.linear(m)

class Decoder(nn.Module):
    def __init__(self, n_feat, n_layer=10, n_embed=1024, n_head=16, condition_dim=512, attn_pdrop=.1, resid_pdrop=.1, mlp_hid_times=4, block_activate='GELU'):
        super(Decoder, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(n_layer):
            self.blocks.append(
                DecoderBlock(
                    n_feat=n_feat,
                    n_embed=n_embed,
                    n_head=n_head,
                    condition_dim=condition_dim,
                    mlp_hid_times=mlp_hid_times,
                    attn_pdrop=attn_pdrop,
                    resid_pdrop=resid_pdrop,
                    activate=block_activate
                )
            )

    def forward(self, x, t, enc, condition=None, padding_masks=None, label_emb=None):
        b, c, _ = x.shape
        mean = []
        for block in self.blocks:
            x, res_mean = block(x, enc, condition, t, padding_masks, label_emb)
            mean.append(res_mean)
        mean = torch.cat(mean, dim=1)
        return x, mean

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(1, max_len, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        """
        # print(x.shape)
        x = x + self.pe
        return self.dropout(x)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class AdaLayerNorm(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.emb = SinusoidalPosEmb(n_embed)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embed, n_embed*2)
        self.layernorm = nn.LayerNorm(n_embed, elementwise_affine=False)

    def forward(self, x, timestep, label_emb=None):
        emb = self.emb(timestep)
        if label_emb is not None:
            emb = emb + label_emb
        emb = self.linear(self.silu(emb)).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x

class FullAttention(nn.Module):
    def __init__(self,
                 n_embed, # the embed dim
                 n_head, # the number of heads
                 attn_pdrop=0.1, # attention dropout prob
                 resid_pdrop=0.1, # residual attention dropout prob
    ):
        super().__init__()
        assert n_embed % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embed, n_embed)
        self.query = nn.Linear(n_embed, n_embed)
        self.value = nn.Linear(n_embed, n_embed)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embed, n_embed)
        self.n_head = n_head

    def forward(self, x, mask=None):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B, nh, T, T)

        att = F.softmax(att, dim=-1) # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False) # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att


class CrossAttention(nn.Module):
    def __init__(self,
                 n_embed,  # the embed dim
                 condition_embed,  # condition dim
                 n_head,  # the number of heads
                 attn_pdrop=0.1,  # attention dropout prob
                 resid_pdrop=0.1,  # residual attention dropout prob
                 ):
        super().__init__()
        assert n_embed % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(condition_embed, n_embed)
        self.query = nn.Linear(n_embed, n_embed)
        self.value = nn.Linear(condition_embed, n_embed)

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embed, n_embed)
        self.n_head = n_head

    def forward(self, x, encoder_output, mask=None):
        B, T, C = x.size()
        B, T_E, _ = encoder_output.size()
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(encoder_output).view(B, T_E, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)

        att = F.softmax(att, dim=-1)  # (B, nh, T, T)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side, (B, T, C)
        att = att.mean(dim=1, keepdim=False)  # (B, T, T)

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, att