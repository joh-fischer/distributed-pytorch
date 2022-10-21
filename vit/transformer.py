# MIT License Copyright (c) 2022 joh-fischer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import torch
import torch.nn as nn

from vit.attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, dim: int = 64, n_heads: int = 4, dropout: float = 0.):
        """
        Transformer Encoder block as described in https://arxiv.org/abs/2010.11929.

        Args:
            dim: Input dimension of transformer block.
            n_heads: Number of heads for attention.
            dropout: p for dropout layers (default: 0).
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, dim, n_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.norm1(x)
        x = self.attn(x)
        x += residual

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x += residual

        return x


if __name__ == "__main__":
    ipt = torch.randn((8, 17, 64))
    tblock = TransformerBlock()
    print("transformer block in:", ipt.shape)             # torch.Size([8, 17, 64])
    print("transformer block out:", tblock(ipt).shape)    # torch.Size([8, 17, 64])
