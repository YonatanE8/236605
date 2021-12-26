from torch import Tensor, flatten
from torchvision.models.resnet import ResNet as TorchResNet
from torchvision.models.resnet import BasicBlock, Bottleneck
from typing import Type, Any, Callable, Union, List, Optional

import torch.nn as nn


class ResNet(TorchResNet):
    def __init__(
            self,
            in_channels: int,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 10,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(ResNet, self).__init__(
            block=block,
            layers=layers,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer,
        )

        inplanes = 64
        self.entry_layer = nn.Conv2d(
            in_channels,
            inplanes,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=3,
            bias=False,
        )

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.entry_layer(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


def resnet18(
        in_channels: int,
        **kwargs: Any,
) -> ResNet:
    return ResNet(
        in_channels=in_channels,
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        **kwargs,
    )


def resnet34(
        in_channels: int,
        **kwargs: Any,
) -> ResNet:
    return ResNet(
        in_channels=in_channels,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        **kwargs,
    )


def resnet50(
        in_channels: int,
        **kwargs: Any,
) -> ResNet:
    return ResNet(
        in_channels=in_channels,
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        **kwargs,
    )


def resnet101(
        in_channels: int,
        **kwargs: Any,
) -> ResNet:
    return ResNet(
        in_channels=in_channels,
        block=Bottleneck,
        layers=[3, 4, 23, 3],
        **kwargs,
    )
