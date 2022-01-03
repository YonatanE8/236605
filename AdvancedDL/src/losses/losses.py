from torch import nn, Tensor
from abc import ABC, abstractmethod
from typing import Sequence, Union, Dict
from torchmetrics.functional import accuracy
from AdvancedDL.src.utils.defaults import Predictions, Targets

import torch


class LossComponent(nn.Module, ABC):
    """
    An abstract API class for loss models
    """

    def __init__(self):
        super(LossComponent, self).__init__()

    @abstractmethod
    def forward(self, inputs: Union[Dict, Sequence]) -> Tensor:
        """
        The forward logic of the loss class.

        :param inputs: (Dict) Either a dictionary with the predictions from the forward pass and the ground truth
        outputs. The possible keys are specified by the following variables:
            MODELS_TENSOR_PREDICITONS_KEY
            MODELS_SEQUENCE_PREDICITONS_KEY
            GT_TENSOR_PREDICITONS_KEY
            GT_SEQUENCE_PREDICITONS_KEY
            GT_TENSOR_INPUTS_KEY
            GT_SEQUENCE_INPUTS_KEY
            OTHER_KEY

        Which can be found under .../DynamicalSystems/utils/defaults.py

        Or a Sequence of dicts, each where each element is a dict with the above-mentioned structure.

        :return: (Tensor) A scalar loss.
        """

        raise NotImplemented

    def __call__(self, inputs: Union[Dict, Sequence]) -> Tensor:
        return self.forward(inputs=inputs)


class ModuleLoss(LossComponent):
    """
    A LossComponent which takes in a PyTorch loss Module and decompose the inputs according to the module's
    expected API.
    """

    def __init__(self, model: nn.Module):
        """
        The constructor for the ModuleLoss class
        :param model: (PyTorch Module) The loss model, containing the computation logic.
        """

        super().__init__()

        self.model = model

    def forward(self, inputs: Dict) -> Tensor:
        """
        Basically a wrapper around the forward of the inner model, which decompose the inputs to the expected
        structure expected by the PyTorch module.

        :param inputs: (dict) The outputs of the forward pass of the model along with the ground-truth labels.
        :return: (Tensor) A scalar Tensor representing the aggregated loss
        """

        y_pred = inputs[Predictions]
        y = inputs[Targets]

        loss = self.model(y_pred, y)

        return loss


class ContrastiveAccuracy(LossComponent):
    def __init__(self, predictions_key: str, target_key: str):
        super().__init__()
        self._predictions_key = predictions_key
        self._target_key = target_key

    def forward(self, inputs: Union[Dict, Sequence]) -> Tensor:
        targets = inputs[self._target_key]
        predictions = inputs[self._predictions_key]
        predictions = torch.argmax(predictions)
        acc = ((predictions - targets).abs() == 0).sum()
        acc = acc / targets.shape[0]

        return acc


class TopKAccuracy(LossComponent):
    def __init__(self, k: int = 1, num_classes: int = 10):
        super(TopKAccuracy, self).__init__()
        self._k = k
        self._num_classes = num_classes

    def forward(self, inputs: Union[Dict, Sequence]) -> Tensor:
        predictions = inputs[Predictions]
        targets = inputs[Targets]

        acc = accuracy(
            preds=predictions,
            target=targets,
            top_k=self._k,
            num_classes=self._num_classes,
        )

        return acc


class CrossEntropy(LossComponent):
    def __init__(self, predictions_key: str, target_key: str):
        super(CrossEntropy, self).__init__()
        self._predictions_key = predictions_key
        self._target_key = target_key
        self.ce = nn.CrossEntropyLoss()

    def forward(self, inputs: Union[Dict, Sequence]) -> Tensor:
        predictions = inputs[self._predictions_key]
        targets = inputs[self._target_key]
        loss = self.ce(predictions, targets)

        return loss


class BinaryCrossEntropy(LossComponent):
    def __init__(self, predictions_key: str, target_key: str):
        super(BinaryCrossEntropy, self).__init__()
        self._predictions_key = predictions_key
        self._target_key = target_key
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs: Union[Dict, Sequence]) -> Tensor:
        predictions = inputs[self._predictions_key]
        predictions = torch.cat([predictions[:, 0], predictions[:, 1:].reshape(-1)])
        targets = inputs[self._target_key]
        loss = self.bce(predictions, targets.type(torch.double))

        return loss

