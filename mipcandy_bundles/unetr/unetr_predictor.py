from typing import override, Mapping, Any

from mipcandy import Predictor
from torch import nn

from mipcandy_bundles.unetr.unetr import make_unetr


class UNETRPredictor(Predictor):
    in_ch: int = 1
    num_classes: int = 1
    img_size: tuple[int, int, int] = (96, 96, 96)

    @override
    def build_network(self, checkpoint: Mapping[str, Any]) -> nn.Module:
        model = make_unetr(self.in_ch, self.num_classes, self.img_size)
        model.load_state_dict(checkpoint)
        return model