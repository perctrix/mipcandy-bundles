from typing import override

import torch
from mipcandy import SegmentationTrainer
from mipcandy.training import TrainerToolbox
from torch import nn

from mipcandy_bundles.unetpp.unetpp import make_unetpp2d


class UNetPPTrainer(SegmentationTrainer):
    deep_supervision: bool = False

    @override
    def build_network(self, example_shape: tuple[int, ...]) -> nn.Module:
        return make_unetpp2d(example_shape[0], self.num_classes, deep_supervision=self.deep_supervision)

    @override
    def backward(self, images: torch.Tensor, labels: torch.Tensor, toolbox: TrainerToolbox) -> tuple[float, dict[str, float]]:
        outputs = toolbox.model(images)

        if self.deep_supervision:
            total_loss = 0
            for output in outputs:
                loss, _ = toolbox.criterion(output, labels)
                total_loss += loss
            total_loss = total_loss / len(outputs)

            final_output = outputs[-1]
            _, metrics = toolbox.criterion(final_output, labels)
        else:
            total_loss, metrics = toolbox.criterion(outputs, labels)

        total_loss.backward()
        return total_loss.item(), metrics

    @override
    def validate_case(self, image: torch.Tensor, label: torch.Tensor, toolbox: TrainerToolbox) -> tuple[float, dict[str, float], torch.Tensor]:
        image, label = image.unsqueeze(0), label.unsqueeze(0)
        outputs = (toolbox.ema if toolbox.ema else toolbox.model)(image)

        if self.deep_supervision:
            total_loss = 0
            for output in outputs:
                loss, _ = toolbox.criterion(output, label)
                total_loss += loss
            total_loss = total_loss / len(outputs)

            final_output = outputs[-1]
            _, metrics = toolbox.criterion(final_output, label)
            return -total_loss.item(), metrics, final_output.squeeze(0)
        else:
            loss, metrics = toolbox.criterion(outputs, label)
            return -loss.item(), metrics, outputs.squeeze(0)

    @override
    def save_preview(self, image: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, *, quality: float = .75) -> None:
        super().save_preview(image, label, mask, quality=quality)
