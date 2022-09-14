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

from models.vit.embedding import PatchEmbedding
from models.vit.transformer import TransformerBlock


class ViT(nn.Module):
    def __init__(self,
                 patch_size: int = 8, depth: int = 16,
                 image_size: int = 32, in_channels: int = 3, n_classes: int = 10,
                 dim: int = 64, n_heads: int = 4, dropout: float = 0.):
        super().__init__()

        # patch embeddings
        self.embedding = PatchEmbedding(image_size, patch_size, in_channels, dim)

        # transformer encoder
        self.transformer = nn.ModuleList([
            TransformerBlock(dim, n_heads, dropout)
            for _ in range(depth)
        ])

        # classification head
        self.head = nn.Sequential(
            nn.Linear(dim, n_classes)
        )

    def forward(self, x: torch.Tensor):
        x = self.embedding(x)

        for tr in self.transformer:
            x = tr(x)

        # just take the class token
        x = x[:, 0]

        x = self.head(x)

        return x


if __name__ == "__main__":
    ipt = torch.randn((8, 3, 32, 32))
    vit = ViT(depth=8)

    print("image in:", ipt.shape)               # torch.Size([8, 3, 32, 32])
    print("vit out:", vit(ipt).shape)           # torch.Size([8, 10])
