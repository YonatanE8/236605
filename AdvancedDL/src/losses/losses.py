from torch import nn, Tensor
from abc import ABC, abstractmethod
from typing import Sequence, Union, Dict
from AdvancedDL.src.utils.defaults import Inputs, Outputs


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

        y_pred = inputs[Inputs]
        y = inputs[Outputs]

        loss = self.model(y, y_pred)

        return loss
