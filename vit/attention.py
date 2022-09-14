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
import einops


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim: int, dim: int = 64, n_heads: int = 4):
        """
        Multi-Head Attention module as described in Attention Is All
        You Need" (https://arxiv.org/abs/1706.03762).

        Args:
            input_dim: Input dimension.
            dim: Dimension of keys, queries, values.
            n_heads: Number of heads for multi-head attention.
        """
        super().__init__()

        self.dim = dim

        self.to_qkv = nn.Linear(input_dim, dim * n_heads * 3, bias=False)
        self.scale = dim ** -0.5

        self.unify_heads = nn.Linear(dim * n_heads, input_dim)

    def forward(self, x: torch.Tensor):
        qkv = self.to_qkv(x)
        q, k, v = einops.rearrange(qkv, 'b t (qkv h d) -> qkv b h t d', qkv=3, d=self.dim)     # [bs, h, t, d]

        dot = torch.einsum('b h t d, b h k d -> b h t k', q, k) * self.scale     # [bs, h, t, t]

        att = torch.softmax(dot, dim=-1)

        out = torch.einsum('b h d t, b h t v -> b h d v', att, v)
        out = einops.rearrange(out, 'b h t d -> b t (h d)')
        out = self.unify_heads(out)

        return out


if __name__ == "__main__":
    ipt = torch.randn((8, 17, 64))
    mha = MultiHeadAttention(64)

    print("in:", ipt.shape)           # torch.Size([8, 17, 64])
    print("out:", mha(ipt).shape)     # torch.Size([8, 17, 64])
