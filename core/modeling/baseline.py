# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
import torch.nn.functional as F
from torch import nn

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class HaarWaveletHighFrequency(nn.Module):
    """One-level Haar extractor that computes high-frequency maps on RGB channels."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _fuse_rgb_high_frequency(detail):
        # Fuse after the RGB Haar step so the input is not collapsed to grayscale first.
        magnitude = torch.sqrt(detail.pow(2).mean(dim=1, keepdim=True) + 1e-12)
        direction = torch.sign(detail.mean(dim=1, keepdim=True))
        return magnitude * direction

    def forward(self, x):
        height, width = x.shape[-2:]

        pad_h = height % 2
        pad_w = width % 2
        rgb = x
        if pad_h or pad_w:
            rgb = F.pad(rgb, (0, pad_w, 0, pad_h), mode='replicate')

        top_left = rgb[:, :, 0::2, 0::2]
        top_right = rgb[:, :, 0::2, 1::2]
        bottom_left = rgb[:, :, 1::2, 0::2]
        bottom_right = rgb[:, :, 1::2, 1::2]

        lh_rgb = (top_left - top_right + bottom_left - bottom_right) * 0.5
        hl_rgb = (top_left + top_right - bottom_left - bottom_right) * 0.5
        hh_rgb = (top_left - top_right - bottom_left + bottom_right) * 0.5

        lh = self._fuse_rgb_high_frequency(lh_rgb)
        hl = self._fuse_rgb_high_frequency(hl_rgb)
        hh = self._fuse_rgb_high_frequency(hh_rgb)

        target_size = (height, width)
        lh = F.interpolate(lh, size=target_size, mode='bilinear', align_corners=False)
        hl = F.interpolate(hl, size=target_size, mode='bilinear', align_corners=False)
        hh = F.interpolate(hh, size=target_size, mode='bilinear', align_corners=False)

        return torch.cat([x, lh, hl, hh], dim=1), {
            'rgb': x,
            'lh': lh,
            'hl': hl,
            'hh': hh,
        }


class MultiScaleBranchFusion(nn.Module):
    """Fixed 1x1/3x3/5x5 branches with attention or direct-sum fusion."""

    def __init__(self, channels, use_attention):
        super().__init__()
        self.use_attention = use_attention
        self.branch1 = self._make_branch(channels, 1, 0)
        self.branch3 = self._make_branch(channels, 3, 1)
        self.branch5 = self._make_branch(channels, 5, 2)
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels * 3, 3, kernel_size=1, bias=True),
            )
        else:
            self.attention = None

    @staticmethod
    def _make_branch(channels, kernel_size, padding):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, return_visuals=False):
        branch_outputs = [self.branch1(x), self.branch3(x), self.branch5(x)]

        if self.use_attention:
            attention_logits = self.attention(torch.cat(branch_outputs, dim=1))
            weights = torch.softmax(attention_logits, dim=1)
            fused = sum(branch_outputs[idx] * weights[:, idx:idx + 1] for idx in range(3))
        else:
            weights = x.new_full((x.shape[0], 3, 1, 1), 1.0 / 3.0)
            fused = sum(branch_outputs)

        visuals = None
        if return_visuals:
            visuals = {
                'branch_weights': weights.detach(),
                'activation_heatmap': fused.detach().mean(dim=1, keepdim=True),
            }

        return fused, visuals


class Baseline(nn.Module):
    in_planes = 2048

    def __init__(
            self,
            num_classes,
            last_stride,
            model_path,
            neck,
            neck_feat,
            model_name,
            pretrain_choice,
            use_wavelet=True,
            use_multiscale_branch=True,
            use_attention_fusion=True,
            return_visuals=False):
        super(Baseline, self).__init__()
        self.use_wavelet = use_wavelet
        self.use_multiscale_branch = use_multiscale_branch
        self.use_attention_fusion = use_attention_fusion
        self.return_visuals = return_visuals
        self.model_name = model_name
        if model_name != 'resnet50' and (use_wavelet or use_multiscale_branch or use_attention_fusion):
            raise ValueError('MS-ReID incremental modules only support MODEL.NAME=resnet50 in this development round.')

        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 6, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 23, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)  
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=64, 
                              reduction=16,
                              dropout_p=0.2, 
                              last_stride=last_stride)
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.wavelet = HaarWaveletHighFrequency() if self.use_wavelet else None
        if self.use_wavelet:
            self.input_adapter = nn.Conv2d(6, 3, kernel_size=1, bias=False)
            self.input_adapter.apply(weights_init_kaiming)
        else:
            self.input_adapter = None

        if self.use_multiscale_branch:
            self.multiscale_fusion = MultiScaleBranchFusion(
                channels=1024,
                use_attention=self.use_attention_fusion,
            )
            self.multiscale_fusion.apply(weights_init_kaiming)
        else:
            self.multiscale_fusion = None

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)     # new add by luo
            # self.classifier.apply(weights_init_classifier)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def _forward_base_with_layer3_hook(self, x, return_visuals=False):
        visuals = {}

        if self.model_name != 'resnet50':
            return self.base(x), visuals

        if self.use_wavelet:
            x, wavelet_visuals = self.wavelet(x)
            if return_visuals:
                visuals['wavelet'] = {name: value.detach() for name, value in wavelet_visuals.items()}
            x = self.input_adapter(x)
        elif return_visuals:
            visuals['wavelet'] = {'rgb': x.detach(), 'lh': None, 'hl': None, 'hh': None}

        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        x = self.base.layer2(x)
        layer3 = self.base.layer3(x)

        if return_visuals:
            visuals['layer3'] = layer3.detach()

        if self.use_multiscale_branch:
            layer3, fusion_visuals = self.multiscale_fusion(layer3, return_visuals=return_visuals)
            if return_visuals:
                visuals['fusion'] = fusion_visuals

        x = self.base.layer4(layer3)
        return x, visuals

    def _build_output(self, backbone_feat):
        global_feat = self.gap(backbone_feat)  # (b, 2048, 1, 1)
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)  # normalize for angular softmax

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                embedding = feat
            else:
                # print("Test with feature before BN")
                embedding = global_feat
            return embedding, global_feat, feat

    def forward(self, x, return_visuals=None):
        if return_visuals is None:
            return_visuals = self.return_visuals and not self.training

        backbone_feat, visuals = self._forward_base_with_layer3_hook(x, return_visuals=return_visuals)
        output = self._build_output(backbone_feat)

        if self.training:
            return output

        embedding, global_feat, feat = output
        if not return_visuals:
            return embedding

        return {
            'embedding': embedding,
            'global_feat': global_feat,
            'bn_feat': feat,
            'visuals': visuals,
        }

    def extract_with_visuals(self, x):
        was_training = self.training
        self.eval()
        try:
            return self.forward(x, return_visuals=True)
        finally:
            self.train(was_training)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path).state_dict()
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])
