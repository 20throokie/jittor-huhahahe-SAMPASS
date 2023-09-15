# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
import jittor as jt
import jittor.nn as nn
from src.vit import ImageEncoderViT
from src.vit_adapter import ViTAdapter
import numpy as np
from src.ViTFeat import vit_tiny


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(jt.ones(num_channels))
        self.bias = nn.Parameter(jt.zeros(num_channels))
        self.eps = eps

    def execute(self, x: jt.Var) -> jt.Var:
        u = x.mean(1, keepdims=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / jt.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MLP2d(nn.Module):
    def __init__(self, in_dim, inner_dim=4096, out_dim=256):
        super(MLP2d, self).__init__()

        self.linear1 = conv1x1(in_dim, inner_dim)
        self.bn1 = nn.BatchNorm2d(inner_dim)
        self.relu1 = nn.ReLU()
        self.linear2 = conv1x1(inner_dim, out_dim)

    def execute(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.linear2(x)

        return x

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = jt.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                'Dilation > 1 not supported in BasicBlock')
        # Both self.conv1 and self.downsample layers downsample the input
        # when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = jt.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input
        # when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def execute(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Mulpixelattn(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden_mlp = channels
        self.atten = nn.Sequential(
            nn.Conv2d(channels, hidden_mlp, 1),
            nn.BatchNorm2d(hidden_mlp),
            nn.ReLU(),
            nn.Conv2d(hidden_mlp, channels, 1),
            nn.BatchNorm2d(channels, affine=True),
        )
        self.threshold = jt.zeros((1, channels, 1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def execute(self, x):
        x = self.atten(x)
        x = x + self.threshold
        att = jt.sigmoid(x)
        return att


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 zero_init_residual=False,
                 groups=1,
                 widen=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None,
                 normalize=False,
                 output_dim=0,
                 hidden_mlp=0,
                 nmb_prototypes=0,
                 eval_mode=False,
                 train_mode='finetune',
                 num_classes=919,
                 shallow=None):
        super().__init__()
        assert train_mode in ['pretrain', 'pixelattn', 'finetune'], train_mode
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.eval_mode = eval_mode
        self.train_mode = train_mode
        self.padding = nn.ConstantPad2d(1, 0.0)
        self.num_classes = num_classes
        self.shallow = shallow
        if isinstance(self.shallow, int):
            self.shallow = [self.shallow]

        self.inplanes = width_per_group * widen
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError('replace_stride_with_dilation should be None '
                             'or a 3-element tuple, got {}'.format(
                                 replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # change padding 3 -> 2 compared to original torchvision code
        # because added a padding layer
        num_out_filters = width_per_group * widen
        # self.conv1 = nn.Conv2d(3,
        #                        num_out_filters,
        #                        kernel_size=7,
        #                        stride=2,
        #                        padding=2,
        #                        bias=False)
        # self.bn1 = norm_layer(num_out_filters)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.layer1 = self._make_layer(block, num_out_filters, layers[0])
        num_out_filters *= 2
        # self.layer2 = self._make_layer(block,
        #                                num_out_filters,
        #                                layers[1],
        #                                stride=2,
        #                                dilate=replace_stride_with_dilation[0])
        num_out_filters *= 2
        # self.layer3 = self._make_layer(block,
        #                                num_out_filters,
        #                                layers[2],
        #                                stride=2,
        #                                dilate=replace_stride_with_dilation[1])
        num_out_filters *= 2
        # self.layer4 = self._make_layer(block,
        #                                num_out_filters,
        #                                layers[3],
        #                                stride=2,
        #                                dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.backbone = ViTAdapter()

        # normalize output features
        self.l2norm = normalize

        # projection head
        if output_dim == 0:
            self.projection_head = None
            if self.train_mode == 'pretrain':
                self.projection_head_shallow = None
        elif hidden_mlp == 0:
            self.projection_head = nn.Linear(num_out_filters * block.expansion,
                                             output_dim)
            if self.train_mode == 'pretrain':
                if self.shallow is not None:
                    self.projection_head_shallow = nn.ModuleList()
                    for stage in shallow:
                        assert stage < 4
                        self.projection_head_shallow.add_module(
                            f'projection_head_shallow{stage}',
                            nn.Linear(num_out_filters * block.expansion,
                                      output_dim))

        else:
            mlps = [
                nn.Linear(768, hidden_mlp),
                #nn.Linear(num_out_filters * block.expansion, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(),
                nn.Linear(hidden_mlp, output_dim),
                nn.BatchNorm1d(output_dim, affine=False)
            ]
            self.projection_head = nn.Sequential(*mlps)
            if self.train_mode == 'pretrain':
                if self.shallow is not None:
                    self.projection_head_shallow = nn.ModuleList()
                    for stage in shallow:
                        assert stage < 4
                        self.projection_head_shallow.add_module(
                            f'projection_head_shallow{stage}',
                            nn.Sequential(
                                nn.Linear(
                                    num_out_filters * block.expansion // (2 * (4 - stage)),
                                    hidden_mlp), nn.BatchNorm1d(hidden_mlp),
                                nn.ReLU(),
                                nn.Linear(hidden_mlp, output_dim),
                                nn.BatchNorm1d(output_dim, affine=False)))
        if self.train_mode == 'pretrain':
            if self.shallow is not None:
                self.projection_head_pixel_shallow = nn.ModuleList()
                for stage in shallow:
                    assert stage < 4
                    self.projection_head_pixel_shallow.add_module(
                        f'projection_head_pixel{stage}',
                        nn.Sequential(
                            nn.Conv2d(
                                num_out_filters * block.expansion // (2 * (4 - stage)),
                                hidden_mlp,
                                kernel_size=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(hidden_mlp),
                            nn.ReLU(),
                            nn.Conv2d(hidden_mlp,
                                      hidden_mlp,
                                      kernel_size=1,
                                      bias=False),
                            nn.BatchNorm2d(hidden_mlp),
                            nn.ReLU(),
                            nn.Conv2d(
                                hidden_mlp,
                                num_out_filters * block.expansion // (2 * (4 - stage)),
                                kernel_size=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_out_filters * block.expansion //
                                           (2 * (4 - stage)),
                                           affine=False),
                        ))

            # projection for pixel-to-pixel
            self.projection_head_pixel = nn.Sequential(
                nn.Conv2d(768,
                          hidden_mlp,
                          kernel_size=1,
                          bias=False),
                # nn.Conv2d(num_out_filters * block.expansion,
                #           hidden_mlp,
                #           kernel_size=1,
                #           bias=False),
                nn.BatchNorm2d(hidden_mlp),
                nn.ReLU(),
                nn.Conv2d(hidden_mlp, hidden_mlp, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_mlp),
                nn.ReLU(),
                nn.Conv2d(hidden_mlp, output_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(output_dim, affine=False),
            )
            self.predictor_head_pixel = nn.Sequential(
                nn.Conv2d(output_dim, output_dim, 1, bias=False),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(),
                nn.Conv2d(output_dim, output_dim, 1),
            )
            # self.pixpro_projector = MLP2d(in_dim=512, inner_dim=1024, out_dim=256)
            # self.pixpro_projector_k = MLP2d(in_dim=512, inner_dim=1024, out_dim=256)
            # self.encoder_k = ImageEncoderViT()
            # for param_q, param_k in zip(self.backbone.parameters(), self.encoder_k.parameters()):
            #     param_k.data = jt.copy(param_q.data)  # initialize
            #     param_k.requires_grad = False  # not update by gradient
            #
            # for param_q, param_k in zip(self.pixpro_projector.parameters(), self.pixpro_projector_k.parameters()):
            #     param_k.data = jt.copy(param_q.data)
            #     param_k.requires_grad = False
            # self.K = int(63016 / 16 / 4 * 200)
            # self.k = int(63016 / 16 / 4 * 1)
            # self.value_transform = conv1x1(in_planes=256, out_planes=256)

        # prototype layer
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)

        if train_mode == 'pixelattn':
            self.fbg = Mulpixelattn(num_out_filters * block.expansion)
            #self.fbg = Mulpixelattn(256)
        elif train_mode == 'finetune':
            self.last_layer = nn.Conv2d(num_out_filters * block.expansion,
                                        num_classes + 1, 1, 1)

            self.up1 = nn.Sequential(
                nn.Conv2d(num_out_filters * block.expansion, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
            self.up2 = nn.Conv2d(256, num_classes + 1, 1, 1)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros,
        # and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3%
        # according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    @jt.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        _contrast_momentum = 1. - (1. - 0.99) * (np.cos(np.pi * self.k / self.K) + 1) / 2.
        self.k = self.k + 1
        if self.k >= self.K:
            self.k = self.K

        for param_q, param_k in zip(self.backbone.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

        for param_q, param_k in zip(self.pixpro_projector.parameters(), self.pixpro_projector_k.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (1. - _contrast_momentum)

    def featprop(self, feat):
        N, C, H, W = feat.shape

        # Value transformation
        feat_value = self.value_transform(feat)
        feat_value = jt.normalize(feat_value, dim=1)
        feat_value = feat_value.view(N, C, -1)

        # Similarity calculation
        feat = jt.normalize(feat, dim=1)

        # [N, C, H * W]
        feat = feat.view(N, C, -1)

        # [N, H * W, H * W]
        attention = jt.bmm(feat.transpose(1, 2), feat)
        attention = jt.clamp(attention, min_v=0)
        attention = attention ** 2

        # [N, C, H * W]
        feat = jt.bmm(feat_value, attention.transpose(1, 2))

        return feat.view(N, C, H, W)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            ))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                ))

        return nn.Sequential(*layers)

    # def execute_backbone(self, x, avgpool=True, mode=None):
    #     x = self.padding(x)
    #
    #     x = self.conv1(x)
    #     x = self.bn1(x)
    #     x = self.relu(x)
    #     x = self.maxpool(x)
    #     x = self.layer1(x)
    #     c1 = x
    #     x = self.layer2(x)
    #     c2 = x
    #     x = self.layer3(x)
    #     c3 = x
    #     x = self.layer4(x)
    #
    #     if mode == 'cluster':
    #         return x, c3, c2
    #
    #     if self.eval_mode or self.train_mode != 'pretrain':
    #         return x
    #
    #     if avgpool:
    #         x = self.avgpool(x)
    #         x = jt.flatten(x, 1)
    #         return x
    #
    #     return x, c3, c2, c1

    def execute_backbone(self, x, avgpool=True, mode=None):
        x = self.backbone.feat_aggregate(x)

        if self.eval_mode or self.train_mode != 'pretrain':
            return x

        if avgpool:
            x = self.avgpool(x)
            x = jt.flatten(x, 1)
            return x

        return x, None, None, None

    def execute_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = jt.normalize(x, dim=1, p=2)

        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def execute_head_shallow(self, x, stage):
        if (self.projection_head_shallow is not None
                and f'projection_head_shallow{stage}'
                in self.projection_head_shallow.keys()):
            x = self.projection_head_shallow.layers[
                f'projection_head_shallow{stage}'](x)

        if self.l2norm:
            x = jt.normalize(x, dim=1, p=2)

        if self.prototypes is not None:
            return x, nn.matmul_transpose(x, self.prototypes.weight.detach()) 
        return x

    def execute_head_pixel(self, x, gridq, gridk):
        if self.projection_head_pixel is not None:
            x = self.projection_head_pixel(x)

        # grid sample 28 x 28
        grid = jt.concat([gridq, gridk], dim=0)
        x = nn.grid_sample(x, grid, align_corners=False, mode='bilinear')

        return x, self.predictor_head_pixel(x)

    def execute(self, inputs, gridq=None, gridk=None, mode='train'):
        if self.train_mode == "pixelattn" and mode == "train":
            inputs, img, sal = inputs
        # if self.train_mode == 'pretrain' and mode == 'train':
        #     inputs, img1, img2, coord1, coord2 = inputs
        if mode == 'cluster':
            output = self.inference_cluster(inputs)
            return output
        elif mode == 'inference_pixel_attention':
            return self.inference_pixel_attention(inputs)

        if self.train_mode == 'finetune':
            out = self.execute_backbone(inputs)
            return self.last_layer(out)

        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops, last_size = [0], inputs[0].shape[-1]
        for sample in [inp.shape[-1] for inp in inputs]:
            if sample == last_size:
                idx_crops[-1] += 1
            else:
                idx_crops.append(idx_crops[-1] + 1)

        start_idx = 0
        for end_idx in idx_crops:
            _out = self.execute_backbone(jt.concat(
                inputs[start_idx:end_idx]), avgpool=self.train_mode != 'pretrain')
            if start_idx == 0:
                if self.train_mode == 'pixelattn':
                    _out = self.execute_pixel_attention(_out)
                elif self.train_mode == 'pretrain':
                    _out, _c3, _c2, _c1 = _out
                    (
                        embedding_deep_pixel,
                        output_deep_pixel,
                    ) = self.execute_head_pixel(_out, gridq, gridk)
                    _stages = [_c1, _c2, _c3]
                    if self.shallow is not None:
                        output_c = []
                        for i, stage in enumerate(self.shallow):
                            _c = _stages[stage - 1]
                            _out_c = self.projection_head_pixel_shallow.layers[
                                f'projection_head_pixel{stage}'](_c)
                            _out_c = self.avgpool(_out_c)
                            _out_c = jt.flatten(_out_c, 1)
                            output_c.append(_out_c)
                    _out = self.avgpool(_out)
                    _out = jt.flatten(_out, 1)
                output = _out
            else:
                if self.train_mode == 'pixelattn':
                    _out = self.execute_pixel_attention(_out)
                elif self.train_mode == 'pretrain':
                    _out, _, _, _ = _out
                    _out = self.avgpool(_out)
                    _out = jt.flatten(_out, 1)
                output = jt.concat((output, _out))
            start_idx = end_idx

        embedding, output = self.execute_head(output)
        if self.shallow is not None:
            for i, stage in enumerate(self.shallow):
                embedding_c_, output_c_ = self.execute_head_shallow(output_c[i],
                                                                  stage=stage)
                embedding = jt.concat((embedding, embedding_c_))
                output = jt.concat((output, output_c_))
        if self.train_mode == 'pixelattn':
            _, temp = self.inference_pixel_attention(img)
            temp = nn.interpolate(
                temp,
                size=(sal.shape[-2], sal.shape[-1]),
                mode='bilinear'
            )
            sal_loss = criterion(temp, sal)
            return sal_loss, embedding, output
        elif self.train_mode == 'pretrain':
            # feat_1 = self.backbone.feat_aggregate(img1)  # queries: NxC
            # proj_1 = self.pixpro_projector(feat_1)
            # pred_1 = self.featprop(proj_1)
            # pred_1 = jt.normalize(pred_1, dim=1)
            #
            # feat_2 = self.backbone.feat_aggregate(img2)
            # proj_2 = self.pixpro_projector(feat_2)
            # pred_2 = self.featprop(proj_2)
            # pred_2 = jt.normalize(pred_2, dim=1)
            # # compute key features
            # with jt.no_grad():  # no gradient to keys
            #     self._momentum_update_key_encoder()  # update the key encoder
            #
            #     feat_1_ng = self.encoder_k.feat_aggregate(img1)  # keys: NxC
            #     proj_1_ng = self.pixpro_projector_k(feat_1_ng)
            #     proj_1_ng = jt.normalize(proj_1_ng, dim=1)
            #
            #     feat_2_ng = self.encoder_k.feat_aggregate(img2)
            #     proj_2_ng = self.pixpro_projector_k(feat_2_ng)
            #     proj_2_ng = jt.normalize(proj_2_ng, dim=1)
            # pixloss = regression_loss(pred_1, proj_2_ng, coord1, coord2, 0.7) \
            #        + regression_loss(pred_2, proj_1_ng, coord2, coord1, 0.7)
            return embedding, output, embedding_deep_pixel, output_deep_pixel
        return embedding, output

    def execute_pixel_attention(self, out, threshold=None):
        out = nn.interpolate(
            out,
            size=(out.shape[2] * 4, out.shape[3] * 4),
            mode='bilinear')
        out = jt.normalize(out, dim=1, p=2)
        fg = self.fbg(out)
        if threshold is not None:
            fg[fg < threshold] = 0

        out = out * fg
        out = self.avgpool(out)
        out = jt.flatten(out, 1)

        return out

    def inference_cluster(self, x, threshold=None):
        out = self.execute_backbone(x)
        out = nn.interpolate(
            out,
            size=(out.shape[2] * 4, out.shape[3] * 4),
            mode='bilinear')
        nout = jt.normalize(out, dim=1, p=2)
        fg = self.fbg(nout)
        if threshold is not None:
            fg[fg < threshold] = 0

        out = out * fg
        out = self.avgpool(out)
        out = jt.flatten(out, 1)

        return out

    def inference_pixel_attention(self, x):
        out = self.execute_backbone(x) # (1,512,8,8)
        out = nn.interpolate(
            out,
            size=(out.shape[2] * 4, out.shape[3] * 4),
            mode='bilinear') # (1,512,32,32)
        out_ = jt.normalize(out, dim=1, p=2) # (1,512,32,32)
        fg = self.fbg(out_) # (1,512,32,32)
        fg = fg.mean(dim=1, keepdims=True) # (1,1,32,32)

        return out, fg


class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super().__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            setattr(self, 'prototypes' + str(i),
                            nn.Linear(output_dim, k, bias=False))

    def execute(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, 'prototypes' + str(i))(x))
        return out

def regression_loss(q, k, coord_q, coord_k, pos_ratio=0.5):
    """ q, k: N * C * H * W
        coord_q, coord_k: N * 4 (x_upper_left, y_upper_left, x_lower_right, y_lower_right)
    """
    N, C, H, W = q.shape
    # [bs, feat_dim, 49]
    q = q.view(N, C, -1)
    k = k.view(N, C, -1)

    # generate center_coord, width, height
    # [1, 7, 7]
    x_array = jt.arange(0., float(W), dtype=coord_q.dtype).view(1, 1, -1).repeat(1, H, 1)
    y_array = jt.arange(0., float(H), dtype=coord_q.dtype).view(1, -1, 1).repeat(1, 1, W)
    # [bs, 1, 1]
    q_bin_width = ((coord_q[:, 2] - coord_q[:, 0]) / W).view(-1, 1, 1)
    q_bin_height = ((coord_q[:, 3] - coord_q[:, 1]) / H).view(-1, 1, 1)
    k_bin_width = ((coord_k[:, 2] - coord_k[:, 0]) / W).view(-1, 1, 1)
    k_bin_height = ((coord_k[:, 3] - coord_k[:, 1]) / H).view(-1, 1, 1)
    # [bs, 1, 1]
    q_start_x = coord_q[:, 0].view(-1, 1, 1)
    q_start_y = coord_q[:, 1].view(-1, 1, 1)
    k_start_x = coord_k[:, 0].view(-1, 1, 1)
    k_start_y = coord_k[:, 1].view(-1, 1, 1)

    # [bs, 1, 1]
    q_bin_diag = jt.sqrt(q_bin_width ** 2 + q_bin_height ** 2)
    k_bin_diag = jt.sqrt(k_bin_width ** 2 + k_bin_height ** 2)
    max_bin_diag = jt.maximum(q_bin_diag, k_bin_diag)

    # [bs, 7, 7]
    center_q_x = (x_array + 0.5) * q_bin_width + q_start_x
    center_q_y = (y_array + 0.5) * q_bin_height + q_start_y
    center_k_x = (x_array + 0.5) * k_bin_width + k_start_x
    center_k_y = (y_array + 0.5) * k_bin_height + k_start_y

    # [bs, 49, 49]
    dist_center = jt.sqrt((center_q_x.view(-1, H * W, 1) - center_k_x.view(-1, 1, H * W)) ** 2
                             + (center_q_y.view(-1, H * W, 1) - center_k_y.view(-1, 1, H * W)) ** 2) / max_bin_diag
    pos_mask = (dist_center < pos_ratio).float().detach()

    # [bs, 49, 49]
    logit = jt.bmm(q.transpose(1, 2), k)

    loss = (logit * pos_mask).sum(-1).sum(-1) / (pos_mask.sum(-1).sum(-1) + 1e-6)

    return -2 * loss.mean()


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def criterion(input, target):
    bceloss = nn.BCELoss()
    return bceloss(input, target)