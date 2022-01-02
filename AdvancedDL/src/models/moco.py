from torch import nn, Tensor
from AdvancedDL.src.models.fc import MLP
from typing import Sequence, Dict, Callable
from AdvancedDL.src.models.resnet import resnet50, IdentityLayer
from AdvancedDL.src.utils.defaults import Queue, Key, Predictions, Labels

import copy
import torch


class MoCoV2(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
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
            },
            self_training: bool = True,
    ):
        super(MoCoV2, self).__init__()

        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature
        resnet = encoder_builder(
            in_channels=in_channels,
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
        self.register_buffer("queue", torch.randn(in_channels, queue_size))
        self.queue = torch.nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Create the linear layer for later use
        self.linear_mlp = MLP(
            n_layers=1,
            in_channels=10,
            out_channels=10,
            l0_units=4096,
            units_grow_rate=1,
            bias=True,
        )
        self._self_training = self_training

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
        assert self.queue_size % bs == 0

        self.queue[:, ptr:(ptr + bs)] = keys.T
        ptr = (ptr + bs) % self.queue_size

        self.queue_ptr[0] = ptr

    def set_self_training(self, status: bool = False):
        self._self_training = status

    def pre_training_params(self) -> Sequence[nn.Parameter]:
        params = list(self.resnet.parameters())
        params += list(self.mlp.parameters())
        return params

    def linear_params(self) -> Sequence[nn.Parameter]:
        params = list(self.linear_mlp.parameters())
        return params

    def freeze(self):
        for name, param in self.named_parameters():
            if 'linear_mlp' not in name:
                param.requires_grad = False

    def forward(self, inputs: dict) -> Dict[str, Tensor]:
        in_q = inputs[Queue]
        in_k = inputs[Key]

        # Compute queue features
        q = self.resnet_q(in_q)

        if not self._self_training:
            # Linear predictor
            q = self.linear_mlp(q)
            q = torch.nn.functional.normalize(q, dim=1)

            # apply temperature
            logits = (q / self.temperature).type(torch.double)
            labels = torch.ones(logits.shape[0], dtype=torch.long).to(logits.device)

        else:
            # MLP predictor
            q = self.mlp_q(q)
            q = torch.nn.functional.normalize(q, dim=1)

            # Compute key features
            with torch.no_grad():
                self._update_key_encoder()

                if not isinstance(self.resnet_q._norm_layer, IdentityLayer):
                    # Shuffle the entries for batch norm
                    bs = in_q.shape[0]
                    shuffle_indices = torch.randperm(bs)
                    in_k = in_k[shuffle_indices]

                k = self.resnet_k(in_k)
                k = self.mlp_k(k)
                k = torch.nn.functional.normalize(k, dim=1)

                if not isinstance(self.resnet_q._norm_layer, IdentityLayer):
                    # Un-shuffle the entries for batch norm
                    un_shuffled_indices = torch.argsort(shuffle_indices).type(torch.int).to(k.device)
                    k = k[un_shuffled_indices]

            # Compute logits
            positive_logits = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            negative_logits = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

            # Insert the positive logit in a random index
            logits = torch.cat([positive_logits, negative_logits], dim=1)

            # apply temperature
            logits = (logits / self.temperature).type(torch.double)
            labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)

            # dequeue and enqueue
            self._dequeue_and_enqueue(k)

        outputs = {
            Predictions: logits,
            Labels: labels,
        }

        return outputs

    def __call__(self, inputs: dict) -> Dict[str, Tensor]:
        return self.forward(inputs=inputs)

