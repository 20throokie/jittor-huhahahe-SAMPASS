# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""
import math
from functools import partial

# import torch
# import torch.nn as nn
import jittor as jt
from jittor import nn

from utils import trunc_normal_


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + jt.rand(shape, dtype=x.dtype)
    random_tensor = jt.floor(random_tensor)
    #random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def execute(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def execute(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def execute(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def execute(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def execute(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """ Vision Transformer """
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(jt.zeros((1, 1, embed_dim)))
        self.pos_embed = nn.Parameter(jt.zeros((1, num_patches + 1, embed_dim)))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in jt.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = myinterpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return jt.concat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = jt.concat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def execute(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def vit_tiny(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_small(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_base(patch_size=16, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)
        #self.last_layer.weight_g.data.fill_(1)
        #if norm_last_layer:
        #    self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def execute(self, x):
        x = self.mlp(x)
        x = jt.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
        
def myinterpolate(X, size=None, scale_factor=None, mode='bilinear', align_corners=False, tf_mode=False):
    if scale_factor is not None:
        size = [int(X.shape[-2] * scale_factor[0]), int(X.shape[-1] * scale_factor[1])]
    if isinstance(size, int):
        size = (size, size)
    if scale_factor is not None:
        return resize(X, size, mode, align_corners, tf_mode)

def resize(img, size, mode="nearest", align_corners=False, tf_mode=False):
    n, c, h, w = img.shape
    H, W = size
    nid, cid, hid, wid = jt.index((n, c, H, W))
    if align_corners:
        x = hid * ((h - 1) / max(1, H - 1))
        y = wid * ((w - 1) / max(1, W - 1))
    elif mode == "bicubic":
        x = (hid + 0.5) * (h / H) - 0.5
        y = (wid + 0.5) * (w / W) - 0.5
    elif mode == 'nearest':
        x = hid * (h / H)
        y = wid * (w / W)
    elif mode == "area":
        '''
        Area interpolation uses AdaptivePool2D to resize origin images.
        '''
        stride = (h // H, w // W)
        assert stride[0] > 0 and stride[1] > 0
        x, y = jt.meshgrid(jt.arange(0, H, 1), jt.arange(0, W, 1))
        startH = jt.floor(x*h/H).int32()
        endH = jt.ceil((x+1)*h/H).int32()
        maxH = int(jt.max(endH - startH).data)
        startW = jt.floor(y*w/W).int32()
        endW = jt.ceil((y+1)*w/W).int32()
        maxW = int(jt.max(endW - startW).data)
        pixel_count = (endH - startH) * (endW - startW)
        adaptive_output = img.reindex([img.shape[0], img.shape[1], H, W, maxH, maxW], ["i0", "i1", "@e0(i2, i3) + i4", "@e2(i2, i3) + i5"], extras=[startH, endH, startW, endW], overflow_conditions=["i4 >= @e1(i2, i3) - @e0(i2, i3)", "i5 >= @e3(i2, i3) - @e2(i2, i3)"], overflow_value=0)
        adaptive_output = adaptive_output.reduce("sum", [4,5]) / pixel_count[None, None, ...]
        return adaptive_output
    else:
        if (tf_mode):
            x = hid * (h / H)
            if H > h: x = x.clamp(0, h - 1)
            y = wid * (w / W)
            if W > w: y = y.clamp(0, w - 1)
        else:
            x = hid * (h / H) + (h / H * 0.5 - 0.5)
            if H > h: x = x.clamp(0, h - 1)
            y = wid * (w / W) + (w / W * 0.5 - 0.5)
            if W > w: y = y.clamp(0, w - 1)
    return _interpolate(img, x, y, (nid, cid), mode)

def _interpolate(img, x, y, ids, mode):
    if mode == "nearest":
        return img.reindex([*ids, x.floor_int(), y.floor_int()])
    if mode == "bilinear":
        fx, fy = x.floor_int(), y.floor_int()
        cx, cy = fx + 1, fy + 1
        dx, dy = x - fx, y - fy
        a = img.reindex_var([*ids, fx, fy])
        b = img.reindex_var([*ids, cx, fy])
        c = img.reindex_var([*ids, fx, cy])
        d = img.reindex_var([*ids, cx, cy])
        dnx, dny = 1 - dx, 1 - dy
        ab = dx * b + dnx * a
        cd = dx * d + dnx * c
        o = ab * dny + cd * dy
        return o
    if mode=="bicubic": # ugly ver.
        n,c,h,w = img.shape
        fx, fy = x.floor_int(), y.floor_int()
        dix, diy = x - fx, y - fy
        ax, ay = _bicubic(dix+1,-0.75,2), _bicubic(diy+1,-0.75,2)
        bx, by = _bicubic(dix,-0.75,1), _bicubic(diy,-0.75,1)
        cx, cy = _bicubic(1-dix,-0.75,1), _bicubic(1-diy,-0.75,1)
        dx, dy = _bicubic(2-dix,-0.75,2), _bicubic(2-diy,-0.75,2)
        afx, afy = jt.maximum(jt.minimum(fx-1,h-1),0), jt.maximum(jt.minimum(fy-1,w-1),0)
        bfx, bfy = jt.maximum(jt.minimum(fx,h-1),0), jt.maximum(jt.minimum(fy,w-1),0)
        cfx, cfy = jt.maximum(jt.minimum(fx+1,h-1),0), jt.maximum(jt.minimum(fy+1,w-1),0)
        dfx, dfy = jt.maximum(jt.minimum(fx+2,h-1),0), jt.maximum(jt.minimum(fy+2,w-1),0)
        a = ax*(img.reindex_var([*ids,afx,afy])*ay+img.reindex_var([*ids,afx,bfy])*by+img.reindex_var([*ids,afx,cfy])*cy+img.reindex_var([*ids,afx,dfy])*dy)
        b = bx*(img.reindex_var([*ids,bfx,afy])*ay+img.reindex_var([*ids,bfx,bfy])*by+img.reindex_var([*ids,bfx,cfy])*cy+img.reindex_var([*ids,bfx,dfy])*dy)
        c = cx*(img.reindex_var([*ids,cfx,afy])*ay+img.reindex_var([*ids,cfx,bfy])*by+img.reindex_var([*ids,cfx,cfy])*cy+img.reindex_var([*ids,cfx,dfy])*dy)
        d = dx*(img.reindex_var([*ids,dfx,afy])*ay+img.reindex_var([*ids,dfx,bfy])*by+img.reindex_var([*ids,dfx,cfy])*cy+img.reindex_var([*ids,dfx,dfy])*dy)
        o = a + b + c + d
        return o
    raise (f"Not support interpolation mode: {mode}")

def _bicubic(x, a, func):
    # normal ver
    if func == 1:
        return (a+2)*(jt.abs(x)**3)-(a+3)*(x**2)+1
    if func == 2:
        return a*(jt.abs(x)**3)-5*a*(x**2)+8*a*(jt.abs(x))-4*a
    return 0
