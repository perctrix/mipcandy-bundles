from typing import override

from mipcandy import SegmentationTrainer
from torch import nn

from mipcandy_bundles.unetr.unetr import make_unetr


class UNETRTrainer(SegmentationTrainer):
    @override
    def build_network(self, example_shape: tuple[int, ...]) -> nn.Module:
        if len(example_shape) != 4:
            raise ValueError(f"UNETR requires 3D input, got shape {example_shape}")
        return make_unetr(example_shape[0], self.num_classes, example_shape[1:])
