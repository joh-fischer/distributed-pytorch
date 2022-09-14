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


class PatchEmbedding(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int = 3, dim: int = 64):
        """
        Embedding of image as described in https://arxiv.org/abs/2010.11929.
        1. Patchify image and linearly project the flattened patches.
        2. Prepend class token.
        3. Add learnable positional embedding.

        Args:
            image_size: Size of the input image.
            patch_size: Size of the patches. `image_size` must be divisible by `patch_size`.
            in_channels: Number of input channels.
            dim: Embedding dimension.
        """
        super().__init__()
        assert image_size % patch_size == 0, "Image size must by divisible by patch size!"

        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size, bias=False)

        # class token
        self.class_token = nn.Parameter(torch.randn(1, 1, dim))

        # positional embedding
        n_patches = (image_size // patch_size) ** 2
        self.pos_emb = nn.Parameter(torch.randn(n_patches + 1, dim))

    def forward(self, x: torch.Tensor):
        bs = x.shape[0]

        # In the paper they first patchify the image and then linearly map
        # it to a lower dimensional space. This is equivalent to first apply
        # a convolution with kernel_size = stride = patch_size and then
        # reshape it. Both result in: H x W x C -> N x (P² * C).
        x = self.projection(x)
        x = einops.rearrange(x, 'b d h w -> b (h w) d')

        # prepend class token
        batched_class_token = einops.repeat(self.class_token, '1 1 d -> b 1 d', b=bs)
        x = torch.cat((batched_class_token, x), dim=1)

        # add learnable positional embedding
        x += self.pos_emb

        return x


if __name__ == "__main__":
    ipt = torch.randn((8, 3, 32, 32))
    patchify = PatchEmbedding(image_size=32, patch_size=8)

    print("image in:", ipt.shape)               # torch.Size([8, 3, 32, 32])
    print("patch emb:", patchify(ipt).shape)    # torch.Size([8, 17, 64])
