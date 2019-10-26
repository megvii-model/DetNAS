# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict

from torch import nn

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.make_layers import conv_with_kaiming_uniform
from . import fpn as fpn_module
from . import resnet
from . import detnasnet


@registry.BACKBONES.register("R-50-C4")
@registry.BACKBONES.register("R-50-C5")
@registry.BACKBONES.register("R-101-C4")
@registry.BACKBONES.register("R-101-C5")
def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    return model


@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
@registry.BACKBONES.register("R-152-FPN")
def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("R-50-FPN-RETINANET")
@registry.BACKBONES.register("R-101-FPN-RETINANET")
def build_resnet_fpn_p3p7_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model

@registry.BACKBONES.register("DETNAS-COCO-FPN-300M")
@registry.BACKBONES.register("DETNAS-COCO-FPN-1.3G")
@registry.BACKBONES.register("DETNAS-COCO-FPN-3.8G")
@registry.BACKBONES.register("DETNAS-COCO-FPN-300M-search")
@registry.BACKBONES.register("DETNAS-COCO-FPN-1.3G-search")
def build_detnasnet_fpn_backbone(cfg):
    body = detnasnet.ShuffleNetV2DetNAS(cfg)
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    if '300M' in cfg.MODEL.BACKBONE.CONV_BODY:
        in_channels_list = [64, 160, 320, 640,]
    elif '1.3G' in cfg.MODEL.BACKBONE.CONV_BODY:
        in_channels_list = [96, 240, 480, 960,]
    elif '3.8G' in cfg.MODEL.BACKBONE.CONV_BODY:
        in_channels_list = [172, 432, 864, 1728,]
    else:
        raise ValueError("Wrong backbone size.")

    fpn = fpn_module.FPN(
        in_channels_list= in_channels_list,
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU, cfg.MODEL.FPN.USE_SYNCBN
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    if 'search' in cfg.MODEL.BACKBONE.CONV_BODY:
        return body, fpn
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("DETNAS-COCO-RetinaNet-300M")
@registry.BACKBONES.register("DETNAS-COCO-RetinaNet-300M-search")
def build_detnasnet_fpn_p3p7_backbone(cfg):
    body = detnasnet.ShuffleNetV2DetNAS(cfg)
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[0, 160, 320, 640,],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU, cfg.MODEL.FPN.USE_SYNCBN
        ),
        top_blocks=fpn_module.LastLevelP6P7(out_channels, out_channels, cfg.MODEL.RETINANET.P6P7_USE_SYNCBN),
    )
    if 'search' in cfg.MODEL.BACKBONE.CONV_BODY:
        return body, fpn
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
