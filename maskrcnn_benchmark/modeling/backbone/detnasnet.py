import torch.nn as nn
from maskrcnn_benchmark.modeling.backbone.shuffle_blocks import ConvBNReLU, ShuffleNetV2BlockSearched, blocks_key


class ShuffleNetV2DetNAS(nn.Module):
    def __init__(self, cfg):
        super(ShuffleNetV2DetNAS, self).__init__()
        model_size = cfg.MODEL.BACKBONE.CONV_BODY.lstrip('DETNAS-')
        print('Model size is {}.'.format(model_size))

        if 'COCO-FPN-3.8G' in model_size:
            architecture = [0, 0, 3, 1, 2, 1, 0, 2, 0, 3, 1, 2, 3, 3, 2, 0, 2, 1, 1, 3,
                            2, 0, 2, 2, 2, 1, 3, 1, 0, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3]
            stage_repeats = [8, 8, 16, 8]
            stage_out_channels = [-1, 72, 172, 432, 864, 1728, 1728]
        elif 'COCO-FPN-1.3G' in model_size:
            architecture = [0, 0, 3, 1, 2, 1, 0, 2, 0, 3, 1, 2, 3, 3, 2, 0, 2, 1, 1, 3,
                            2, 0, 2, 2, 2, 1, 3, 1, 0, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3]
            stage_repeats = [8, 8, 16, 8]
            stage_out_channels = [-1, 48, 96, 240, 480, 960, 1024]
        elif 'COCO-FPN-300M' in model_size:
            architecture = [2, 1, 2, 0, 2, 1, 1, 2, 3, 3, 1, 3, 0, 0, 3, 1, 3, 1, 3, 2]
            #architecture = [0, 0, 0, 1, 2, 0, 3, 3, 1, 2, 2, 2, 3, 3, 3, 1, 3, 2, 3, 2] # search from tong 1019
            stage_repeats = [4, 4, 8, 4]
            stage_out_channels = [-1, 16, 64, 160, 320, 640, 1024]
        elif 'COCO-RetinaNet-300M' in model_size:
            architecture = [2, 3, 1, 1, 3, 2, 1, 3, 3, 1, 1, 1, 3, 3, 2, 0, 3, 3, 3, 3]
            stage_repeats = [4, 4, 8, 4]
            stage_out_channels = [-1, 16, 64, 160, 320, 640, 1024]
        else:
            raise NotImplementedError

        if 'search' in model_size:
            architecture = None
            self.blocks_key = blocks_key
            self.num_states = sum(stage_repeats)

        self.first_conv = ConvBNReLU(in_channel=3, out_channel=stage_out_channels[1], k_size=3, stride=2, padding=1, gaussian_init=True)

        self.features = list()
        self.stage_ends_idx = list()

        in_channels = stage_out_channels[1]
        i_th = 0
        for id_stage in range(1, len(stage_repeats) + 1):
            out_channels = stage_out_channels[id_stage + 1]
            repeats = stage_repeats[id_stage - 1]
            for id_repeat in range(repeats):
                prefix = str(id_stage) + chr(ord('a') + id_repeat)
                stride = 1 if id_repeat > 0 else 2
                if architecture is None:
                    _ops = nn.ModuleList()
                    for i in range(len(blocks_key)):
                        _ops.append(ShuffleNetV2BlockSearched(prefix, in_channels=in_channels, out_channels=out_channels,
                                                               stride=stride, base_mid_channels=out_channels // 2, id=i))
                    self.features.append(_ops)
                else:
                    self.features.append(ShuffleNetV2BlockSearched(prefix, in_channels=in_channels, out_channels=out_channels,
                                                               stride=stride, base_mid_channels=out_channels // 2, id=architecture[i_th]))
                in_channels = out_channels
                i_th += 1
            self.stage_ends_idx.append(i_th-1)

        self.features = nn.Sequential(*self.features)

    def forward(self, x, rngs=None):
        outputs = []
        x = self.first_conv(x)

        for i, select_op in enumerate(self.features):
            x = select_op(x) if rngs is None else select_op[rngs[i]](x)
            if i in self.stage_ends_idx:
                outputs.append(x)
        return outputs


if __name__ == "__main__":
    from maskrcnn_benchmark.config import cfg
    model = ShuffleNetV2DetNAS(cfg)
    print(model)
