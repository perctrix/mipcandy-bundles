import torch
from mipcandy import LayerT
from torch import nn
from typing import Literal, Sequence

from mipcandy_bundles.unet.unet import UNetDoubleConv, UNetOut


class UNetPP(nn.Module):

    def _get_name(self):
        return "UNet++"
    
    def __init__(self, in_ch: int, num_classes: int, hidden_chs: Sequence[int], *,
                 deep_supervision: bool = False, num_dims: Literal[2, 3] = 2,
                 linear: bool = True,
                 conv: LayerT = LayerT(nn.Conv2d),
                 norm: LayerT = LayerT(nn.InstanceNorm2d, num_features="in_ch", affine=True),
                 act: LayerT = LayerT(nn.ReLU, inplace=True),
                 conv_block: LayerT = LayerT(UNetDoubleConv),
                 pool: LayerT = LayerT(nn.MaxPool2d),
                 bias: bool = True) -> None:
        super().__init__()

        self.hidden_chs: Sequence[int] = hidden_chs
        self.deep_supervision: bool = deep_supervision
        self.num_layers: int = len(hidden_chs) - 1

        if num_dims == 2:
            transpose_conv = LayerT(nn.ConvTranspose2d)
            upsample_mode = "bilinear"
        else:
            transpose_conv = LayerT(nn.ConvTranspose3d)
            upsample_mode = "trilinear"

        self.pool: nn.Module = pool.assemble(2, 2)
        self.linear = linear
        if linear:
            self.up: nn.Module = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=True)
        else:
            self.transpose_conv = transpose_conv

        self.convs: nn.ModuleList = nn.ModuleList()
        for i in range(self.num_layers + 1):
            layer_convs = nn.ModuleList()
            for j in range(self.num_layers - i + 1):
                match (i, j):
                    case (0, 0):
                        in_channels = in_ch
                        out_channels = hidden_chs[0]
                    case (_, 0):
                        in_channels = hidden_chs[i-1]
                        out_channels = hidden_chs[i]
                    case _:
                        in_channels = hidden_chs[i] * j + hidden_chs[i+1]
                        out_channels = hidden_chs[i]

                layer_convs.append(conv_block.assemble(
                    in_channels, out_channels, mid_ch=out_channels,
                    conv=conv, norm=norm, act=act, bias=bias
                ))
            self.convs.append(layer_convs)

        if self.deep_supervision:
            self.finals: nn.ModuleList = nn.ModuleList()
            for j in range(1, self.num_layers + 1):
                self.finals.append(UNetOut(hidden_chs[0], num_classes, conv=conv))
        else:
            self.final: nn.Module = UNetOut(hidden_chs[0], num_classes, conv=conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        cache = []
        for i in range(self.num_layers + 1):
            cache.append([None] * (self.num_layers - i + 1))

        cache[0][0] = self.convs[0][0](x)
        for i in range(1, self.num_layers + 1):
            cache[i][0] = self.convs[i][0](self.pool(cache[i-1][0]))

        for j in range(1, self.num_layers + 1):
            for i in range(self.num_layers - j + 1):
                inputs: list[torch.Tensor] = []
                for k in range(j):
                    inputs.append(cache[i][k])

                if self.linear:
                    upsampled = self.up(cache[i+1][j-1])
                else:
                    in_ch = cache[i+1][j-1].shape[1]
                    out_ch = in_ch // 2
                    transpose_conv = self.transpose_conv.assemble(in_ch, out_ch, kernel_size=2, stride=2)
                    upsampled = transpose_conv(cache[i+1][j-1])

                inputs.append(upsampled)
                cache[i][j] = self.convs[i][j](torch.cat(inputs, dim=1))

        if self.deep_supervision and self.training:
            outputs = []
            for j in range(1, self.num_layers + 1):
                outputs.append(self.finals[j-1](cache[0][j]))
            return outputs
        else:
            return self.final(cache[0][self.num_layers])


def make_unetpp2d(in_ch: int, num_classes: int, *,
                  hidden_chs: Sequence[int] = (32, 64, 128, 256, 512, 512, 512, 512),
                  deep_supervision: bool = False, linear: bool = True) -> UNetPP:
    return UNetPP(in_ch, num_classes, hidden_chs, deep_supervision=deep_supervision,
                  linear=linear, norm=LayerT(nn.BatchNorm2d, num_features="in_ch"))


def make_unetpp3d(in_ch: int, num_classes: int, *,
                  hidden_chs: Sequence[int] = (32, 64, 128, 256, 320),
                  deep_supervision: bool = False, linear: bool = True) -> UNetPP:
    return UNetPP(in_ch, num_classes, hidden_chs, deep_supervision=deep_supervision,
                  num_dims=3, conv=LayerT(nn.Conv3d),
                  norm=LayerT(nn.BatchNorm3d, num_features="in_ch"),
                  pool=LayerT(nn.MaxPool3d), linear=linear)

if __name__ == '__main__':
    from mipcandy import sanity_check

    print("Testing UNet++ 2D...")
    model = make_unetpp2d(3, 1)
    results = sanity_check(model, input_shape=(3, 256, 256))
    print(results)
    print(results.output.shape)

    print("\nTesting UNet++ 3D...")
    model = make_unetpp3d(4, 1)
    results = sanity_check(model, input_shape=(4, 64, 192, 192), device='cpu')
    print(results)
    print(results.output.shape)
