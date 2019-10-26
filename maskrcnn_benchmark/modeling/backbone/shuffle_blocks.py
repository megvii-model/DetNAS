import torch
import torch.nn as nn
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.pytorch_distributed_syncbn.syncbn import DistributedSyncBN

batch_norm = DistributedSyncBN

blocks_key = [
    'shufflenet_3x3',
    'shufflenet_5x5',
    'shufflenet_7x7',
    'xception_3x3',
]


Blocks = {
  'shufflenet_3x3': lambda prefix, in_channels, output_channels, base_mid_channels, stride, bn_training: conv1x1_dwconv_conv1x1(prefix, in_channels, output_channels, base_mid_channels, 3, stride, bn_training),
  'shufflenet_5x5': lambda prefix, in_channels, output_channels, base_mid_channels, stride, bn_training: conv1x1_dwconv_conv1x1(prefix, in_channels, output_channels, base_mid_channels, 5, stride, bn_training),
  'shufflenet_7x7': lambda prefix, in_channels, output_channels, base_mid_channels, stride, bn_training: conv1x1_dwconv_conv1x1(prefix, in_channels, output_channels, base_mid_channels, 7, stride, bn_training),
  'xception_3x3': lambda prefix, in_channels, output_channels, base_mid_channels, stride, bn_training: xception(prefix, in_channels, output_channels, base_mid_channels, stride, bn_training),
}


def create_spatial_conv2d_group_bn_relu(prefix, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1, groups=1,
                          bias=False, has_bn=True, has_relu=True, channel_shuffle=False, has_spatial_conv=True, has_spatial_conv_bn=True,
                          conv_name_fun=None, bn_name_fun=None, bn_training=True, fix_weights=False):
    conv_name = prefix
    if conv_name_fun:
        conv_name = conv_name_fun(prefix)

    layer = nn.Sequential()

    if has_spatial_conv:
        spatial_conv_name = conv_name + '_s'
        layer.add_module(spatial_conv_name, nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                                      kernel_size=kernel_size, stride=stride, padding=padding,
                                                      dilation=dilation, groups=in_channels, bias=bias))
        if fix_weights:
            pass

        if has_spatial_conv_bn:
            layer.add_module(spatial_conv_name + '_bn', batch_norm(in_channels))

    if channel_shuffle:
        pass

    assert in_channels % groups == 0
    assert out_channels % groups == 0

    layer.add_module(conv_name, nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                    kernel_size=1, stride=1, padding=0,
                                                    groups=groups, bias=bias))
    if fix_weights:
        pass

    if has_bn:
        bn_name = 'bn_' + prefix
        if bn_name_fun:
            bn_name = bn_name_fun(prefix)
        layer.add_module(bn_name, batch_norm(out_channels))
        if bn_training:
            pass

    if has_relu:
        layer.add_module('relu' + prefix, nn.ReLU(inplace=False)) #True))

    return layer


def conv1x1_dwconv_conv1x1(prefix, in_channels, out_channels, mid_channels, kernel_size, stride, bn_training=True):
    mid_channels = int(mid_channels)
    layer = list()

    layer.append(create_spatial_conv2d_group_bn_relu(prefix=prefix + '_branch2a', in_channels=in_channels, out_channels=mid_channels,
                                                     kernel_size=-1, stride=1, padding=0, groups=1, has_bn=True, has_relu=True,
                                                     channel_shuffle=False, has_spatial_conv=False, has_spatial_conv_bn=False,
                                                     conv_name_fun=lambda p: 'interstellar' + p,
                                                     bn_name_fun=lambda p: 'bn' + p,
                                                     bn_training=bn_training))
    layer.append(create_spatial_conv2d_group_bn_relu(prefix=prefix + '_branch2b', in_channels=mid_channels, out_channels=out_channels,
                                                     kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=1,
                                                     has_bn=True, has_relu=False, channel_shuffle=False, has_spatial_conv=True,
                                                     has_spatial_conv_bn=True,
                                                     conv_name_fun=lambda p: 'interstellar' + p,
                                                     bn_name_fun=lambda p: 'bn' + p,
                                                     bn_training=bn_training))
    return nn.Sequential(*layer)


def xception(prefix, in_channels, out_channels, mid_channels, stride, bn_training=True):
    mid_channels = int(mid_channels)
    layer = list()

    layer.append(create_spatial_conv2d_group_bn_relu(prefix=prefix + '_branch2a', in_channels=in_channels, out_channels=mid_channels,
                                                     kernel_size=3, stride=stride, padding=1, groups=1, has_bn=True, has_relu=True,
                                                     channel_shuffle=False, has_spatial_conv=True, has_spatial_conv_bn=True,
                                                     conv_name_fun=lambda p: 'interstellar' + p,
                                                     bn_name_fun=lambda p: 'bn' + p,
                                                     bn_training=bn_training))

    layer.append(create_spatial_conv2d_group_bn_relu(prefix=prefix + '_branch2b', in_channels=mid_channels,
                                                     out_channels=mid_channels,
                                                     kernel_size=3, stride=1, padding=1, groups=1, has_bn=True,
                                                     has_relu=True,
                                                     channel_shuffle=False, has_spatial_conv=True,
                                                     has_spatial_conv_bn=True,
                                                     conv_name_fun=lambda p: 'interstellar' + p,
                                                     bn_name_fun=lambda p: 'bn' + p,
                                                     bn_training=bn_training))

    layer.append(create_spatial_conv2d_group_bn_relu(prefix=prefix + '_branch2c', in_channels=mid_channels,
                                                     out_channels=out_channels,
                                                     kernel_size=3, stride=1, padding=1, groups=1, has_bn=True,
                                                     has_relu=False,
                                                     channel_shuffle=False, has_spatial_conv=True,
                                                     has_spatial_conv_bn=True,
                                                     conv_name_fun=lambda p: 'interstellar' + p,
                                                     bn_name_fun=lambda p: 'bn' + p,
                                                     bn_training=bn_training))
    return nn.Sequential(*layer)


class ConvBNReLU(nn.Module):

    def __init__(self, in_channel, out_channel, k_size, stride=1, padding=0, groups=1,
                 has_bn=True, has_relu=True, gaussian_init=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=k_size,
                              stride=stride, padding=padding,
                              groups=groups, bias=True)
        if gaussian_init:
            nn.init.normal_(self.conv.weight.data, 0, 0.01)

        if has_bn:
            self.bn = batch_norm(out_channel)

        self.has_bn = has_bn
        self.has_relu = has_relu
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x


def channel_shuffle1(x):
    groups = 2
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    x = x.view(batchsize, -1, height, width)
    x1 = x[:, :(x.shape[1] // 2), :, :]
    x2 = x[:, (x.shape[1] // 2):, :, :]
    return x1, x2


def channel_shuffle2(x):
    channels = x.shape[1]
    assert channels % 4 == 0

    height = x.shape[2]
    width = x.shape[3]

    x = x.reshape(x.shape[0] * channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, channels // 2, height, width)
    return x[0], x[1]


class ShuffleNetV2BlockSearched(nn.Module):
    def __init__(self, prefix, in_channels, out_channels, stride, base_mid_channels, id):
        super(ShuffleNetV2BlockSearched, self).__init__()
        op = blocks_key[id]
        self.ksize = int(op.split('_')[1][0])
        self.stride = stride
        if self.stride == 2:
            self.conv = Blocks[op](prefix + '_' + op, in_channels, out_channels - in_channels, base_mid_channels, stride, True)
        else:
            self.conv = Blocks[op](prefix + '_' + op, in_channels // 2, out_channels // 2, base_mid_channels, stride, True)
        if stride > 1:
            self.proj_conv = create_spatial_conv2d_group_bn_relu(prefix + '_proj', in_channels, in_channels, self.ksize,
                                                                 stride, self.ksize // 2,
                                                                 has_bn=True, has_relu=True, channel_shuffle=False,
                                                                 has_spatial_conv=True, has_spatial_conv_bn=True,
                                                                 conv_name_fun=lambda p: 'interstellar' + p,
                                                                 bn_name_fun=lambda p: 'bn' + p)
        self.relu = nn.ReLU(inplace=False)
        self.channel_shuffle = channel_shuffle1 if cfg.MODEL.BACKBONE.CHANNEL_SHUFFLE_METHOD==1 else channel_shuffle2

    def forward(self, x_in):
        if self.stride == 1:
            x_proj, x = self.channel_shuffle(x_in)
        else:
            x_proj = x_in
            x = x_in
            x_proj = self.proj_conv(x_proj)
        x = self.relu(self.conv(x))

        return torch.cat((x_proj, x), dim=1)


