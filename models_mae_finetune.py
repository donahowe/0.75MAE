# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import copy
import numpy as np
import torch
import torch.nn as nn
import statistics
from random import sample
from torch.nn import functional as F


from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed

def cross_entropy(input, target, weight=None, reduction='mean', ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear', align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)

def conv_diff(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU()
    )




class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvLayer, self).__init__()
        #         reflection_padding = kernel_size // 2
        #         self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        #         out = self.reflection_pad(x)
        out = self.conv2d(x)
        return out


def make_prediction(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    )


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self,img_size=256,
                 patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,embedding_dim= 4, output_nc=2,decoder_softmax = False):
        super().__init__()
       
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.decoder_embed_dim=decoder_embed_dim
        self.decoder_depth=decoder_depth
        self.decoder_num_heads=decoder_num_heads
        self.mlp_ratio=mlp_ratio
        self.norm_layer=norm_layer
        self.norm_pix_loss=norm_pix_loss
        self.embedding_dim=embedding_dim
        self.output_nc=output_nc


        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics

        self.diff_c   = conv_diff(in_channels=2*self.embedding_dim, out_channels=self.embedding_dim)

        self.change_probability = ConvLayer(self.embedding_dim, self.output_nc, kernel_size=3, stride=1, padding=1)

        #Final activation
        self.output_softmax     = decoder_softmax
        self.active             = nn.Sigmoid()

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.

        # initialize nn.Linear and nn.LayerNorm

        # model = MaskedAutoencoderViT(img_size=256, in_chans=3,
        #                              patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        #                              decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 4))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 4, h * p, h * p))
        return imgs



    def forward_encoder(self, x, x_1):
        # embed patches

        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x_1 = self.patch_embed(x_1)
        x_1 = x_1 + self.pos_embed[:, 1:, :]

        for blk in self.blocks:
            x = blk(x)
            x_1 = blk(x_1)
        x = self.norm(x)
        x_1 = self.norm(x_1)
        x = self.unpatchify(x)
        x_1 = self.unpatchify(x_1)
        return x, x_1


    def forward_decoder(self, inputs1, inputs2):
        _c   = self.diff_c(torch.cat((inputs1, inputs2), dim=1))

        #Final prediction
        cp = self.change_probability(_c)
        outputs = cp
        if self.output_softmax:
            outputs=self.active(cp)
        return outputs


    def forward(self, imgs, image_t2, lable):
        [fx1, fx2] = self.forward_encoder(imgs, image_t2)

        cp = self.forward_decoder(fx1, fx2)

        # # Save to mat
        # save_to_mat(x1, x2, fx1, fx2, cp, "ChangeFormerV4")

        # exit()
        acc = cross_entropy(cp, lable)
        return cp, acc






def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(img_size=256,
        patch_size=16, in_chans=3,
        embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4., norm_layer=nn.LayerNorm, embedding_dim=4, output_nc=2, decoder_softmax=False,**kwargs)
    return model




mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks

