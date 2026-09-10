"""Vision Transformer model for CIFAR-style inputs.

Adapted from:
https://github.com/dqj5182/ViT-PyTorch
(original file path: model/vit/vit.py)
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_chans: int = 3, emb_dim: int = 48):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_chans,
            emb_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.proj(x)
            x = x.flatten(2)
            x = x.transpose(1, 2)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, mlp_hidden_dim: int, num_head: int = 8, dropout: float = 0.0):
        super(TransformerEncoder, self).__init__()
        self.norm1 = nn.LayerNorm(input_dim)
        self.msa = MultiHeadSelfAttention(input_dim, n_heads=num_head)
        self.norm2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, input_dim),
            nn.GELU(),
        )

    def forward(self, x):
        out = self.msa(self.norm1(x)) + x
        out = self.mlp(self.norm2(out)) + out
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int = 8, qkv_bias: bool = True, attn_p: float = 0.01, proj_p: float = 0.01):
        super(MultiHeadSelfAttention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        batch_size, n_tokens, x_dim = x.shape

        if x_dim != self.dim:
            raise ValueError
        if self.dim != self.head_dim * self.n_heads:
            raise ValueError("Input & output dim must be divisible by number of heads")

        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, n_tokens, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        k_t = k.transpose(-2, -1)
        dot_product = (q @ k_t) * self.scale
        attn = dot_product.softmax(dim=-1)
        attn = self.attn_drop(attn)
        weighted_avg = attn @ v
        weighted_avg = weighted_avg.transpose(1, 2)

        weighted_avg = weighted_avg.flatten(2)
        x = self.proj(weighted_avg)
        x = self.proj_drop(x)

        return x


class ViT(nn.Module):
    def __init__(
        self,
        in_c: int = 3,
        num_classes: int = 10,
        img_size: int = 32,
        num_patch_1d: int = 8,
        dropout: float = 0.0,
        num_enc_layers: int = 7,
        hidden_dim: int = 384,
        mlp_hidden_dim: int = 384 * 4,
        num_head: int = 8,
        is_cls_token: bool = True,
    ):
        super(ViT, self).__init__()
        self.is_cls_token = is_cls_token
        self.num_patch_1d = num_patch_1d
        self.patch_size = img_size // self.num_patch_1d
        flattened_patch_dim = (img_size // self.num_patch_1d) ** 2 * 3
        num_tokens = (self.num_patch_1d ** 2) + 1 if self.is_cls_token else (self.num_patch_1d ** 2)

        self.images_to_patches = PatchEmbedding(
            img_size=img_size,
            patch_size=img_size // num_patch_1d,
        )

        self.lpfp = nn.Linear(flattened_patch_dim, hidden_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim)) if is_cls_token else None
        self.pos_emb = nn.Parameter(torch.randn(1, num_tokens, hidden_dim))

        enc_list = [
            TransformerEncoder(hidden_dim, mlp_hidden_dim=mlp_hidden_dim, dropout=dropout, num_head=num_head)
            for _ in range(num_enc_layers)
        ]
        self.enc = nn.Sequential(*enc_list)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def forward(self, x):
        out = self.images_to_patches(x)
        out = self.lpfp(out)

        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0), 1, 1), out], dim=1)
        out = out + self.pos_emb
        out = self.enc(out)
        if self.is_cls_token:
            out = out[:, 0]
        else:
            out = out.mean(1)

        out = self.mlp_head(out)
        return out

    def _init_weights(self):
        nn.init.xavier_uniform_(self.lpfp.weight)
        if self.lpfp.bias is not None:
            nn.init.zeros_(self.lpfp.bias)

        nn.init.xavier_uniform_(self.mlp_head[1].weight)
        if self.mlp_head[1].bias is not None:
            nn.init.zeros_(self.mlp_head[1].bias)

        nn.init.normal_(self.pos_emb, std=0.02)
