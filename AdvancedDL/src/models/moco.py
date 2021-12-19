from torch import nn, Tensor, functional
from typing import Sequence, Dict, Callable
from AdvancedDL.src.models.fc import MLP
from AdvancedDL.src.models.resnet import resnet50
from AdvancedDL.src.utils.defaults import Queue, Key, Logits, Predictions

import copy
import torch


class MoCoV2(nn.Module):
    def __init__(
            self,
            input_dim: int,
            encoder_builder: Callable = resnet50,
            queue_size: int = 65536,
            momentum: float = 0.999,
            temperature: float = 0.2,
            resnet_kwargs: dict = {},
            mlp_params: dict = {
                'n_layers': 3,
                'in_channels': 1,
                'out_channels': 10,
                'l0_units': 2048,
                'units_grow_rate': 1,
                'bias': True,
                'activation': 'relu',
            }
    ):
        super(MoCoV2, self).__init__()

        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        resnet = encoder_builder(
            input_dim=input_dim,
            **resnet_kwargs
        )
        mlp = MLP(
            **mlp_params
        )

        # Create the v2 encoders
        self.resnet_q = copy.deepcopy(resnet)
        self.resnet_k = copy.deepcopy(resnet)
        self.mlp_q = copy.deepcopy(mlp)
        self.mlp_k = copy.deepcopy(mlp)

        for param in self.resnet_k.parameters():
            param.requires_grad = False

        for param in self.mlp_k.parameters():
            param.requires_grad = False

        # Create the queue
        self.register_buffer("queue", torch.randn(input_dim, queue_size))
        self.queue = functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Create the linear layer for later use
        self.linear_mlp = MLP(
            n_layers=1,
            in_channels=1,
            out_channels=10,
            l0_units=4096,
            units_grow_rate=1,
            bias=True,
        )
        self.mode = 0

    @torch.no_grad()
    def _update_key_encoder(self):
        for param_k, param_q in zip(self.resnet_k.parameters(), self.resnet_q.parameters()):
            param_k.data = (self.momentum * param_k.data) + (param_q.data * (1 - self.momentum))

        for param_k, param_q in zip(self.mlp_k.parameters(), self.mlp_q.parameters()):
            param_k.data = (self.momentum * param_k.data) + (param_q.data * (1 - self.momentum))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: Tensor):
        bs = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % bs == 0

        self.queue[:, ptr:(ptr + bs)] = keys.T
        ptr = (ptr + bs) % self.K

        self.queue_ptr[0] = ptr

    def update_mode(self, mode: int = 0):
        self.mode = mode

    def pre_training_params(self) -> Sequence[nn.Parameter]:
        params = list(self.resnet.parameters())
        params += list(self.mlp.parameters())
        return params

    def linear_params(self) -> Sequence[nn.Parameter]:
        params = list(self.linear_mlp.parameters())
        return params

    def forward(self, inputs: dict) -> Dict[str, Tensor]:
        in_q = inputs[Queue]
        in_k = inputs[Key]

        # Compute queue features
        q = self.resnet_q(in_q)
        q = self.mlp_q(q)
        q = functional.normalize(q, dim=1)

        # Compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()

            k = self.resnet_k(in_k)
            k = self.mlp_k(k)
            k = functional.normalize(k, dim=1)

        # Compute logits
        positive_logits = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        negative_logits = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        logits = torch.cat([positive_logits, negative_logits], dim=1)

        # apply temperature
        logits = logits / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        outputs = {
            Logits: logits,
            Predictions: labels,
        }

        return outputs
