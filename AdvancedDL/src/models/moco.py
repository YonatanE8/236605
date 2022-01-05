from torch import nn, Tensor
from typing import Dict, Callable
from AdvancedDL.src.models import resnet as resnets
from AdvancedDL.src.utils.defaults import Queue, Key, Predictions, Labels

import torch


class MoCoV2(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            encoder_builder: Callable = resnets.resnet50,
            queue_size: int = 65536,
            momentum: float = 0.999,
            temperature: float = 0.2,
            resnet_kwargs: dict = {},
            self_training: bool = True,
    ):
        super(MoCoV2, self).__init__()

        self.queue_size = queue_size
        self.momentum = momentum
        self.temperature = temperature

        # Create the ResNet back-bone with non-linear MLP
        if encoder_builder == resnets.resnet50 or encoder_builder == resnets.resnet101:
            in_features = 2048
            out_features = 4096
            self._out_feature_dim = 2048

        elif encoder_builder == resnets.resnet18 or encoder_builder == resnets.resnet34:
            in_features = 512
            out_features = 1024
            self._out_feature_dim = 512

        resnet = encoder_builder(
            in_channels=in_channels,
            **resnet_kwargs
        )
        mlp = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=False),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=out_features, out_features=self._out_feature_dim, bias=True),
        )
        resnet.fc = mlp

        self.resnet_q = resnet

        resnet = encoder_builder(
            in_channels=in_channels,
            **resnet_kwargs
        )
        mlp = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features, bias=False),
            nn.BatchNorm1d(out_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=out_features, out_features=self._out_feature_dim, bias=True),
        )
        resnet.fc = mlp
        self.resnet_k = resnet

        for param in self.resnet_k.parameters():
            param.requires_grad = False

        # Create the queue
        self.register_buffer("queue", torch.randn(self._out_feature_dim, queue_size))
        self.queue = torch.nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Create the linear layer for later use
        self.linear_mlp = nn.Linear(in_features=self._out_feature_dim, out_features=10, bias=True)

        self._self_training = self_training

    @torch.no_grad()
    def _update_key_encoder(self):
        for param_k, param_q in zip(self.resnet_k.parameters(), self.resnet_q.parameters()):
            param_k.data = (self.momentum * param_k.data) + (param_q.data.clone() * (1 - self.momentum))

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

    def freeze(self):
        for param in self.resnet_q.parameters():
            param.requires_grad = False

    def forward(self, inputs: dict) -> Dict[str, Tensor]:
        in_q = inputs[Queue]
        in_k = inputs[Key]
        batch_size = in_q.shape[0]

        # Compute queue features
        q = self.resnet_q(in_q)
        q = torch.nn.functional.normalize(q, dim=1)

        if self._self_training:
            with torch.no_grad():
                self._update_key_encoder()

                # Shuffle the entries for batch norm
                bs = in_q.shape[0]
                shuffle_indices = torch.randperm(bs)
                in_k = in_k[shuffle_indices]

                # Compute key features
                k = self.resnet_k(in_k)
                k = torch.nn.functional.normalize(k, dim=1)

                # Un-shuffle the entries for batch norm
                un_shuffled_indices = torch.argsort(shuffle_indices).to(k.device)
                k = k[un_shuffled_indices]

                # dequeue and enqueue
                if self.training:
                    self._dequeue_and_enqueue(k)

            # Compute logits
            positive_logits = torch.bmm(
                q.view(batch_size, 1, self._out_feature_dim),
                k.view(batch_size, self._out_feature_dim, 1)
            )
            negative_logits = torch.mm(
                q.view(batch_size, self._out_feature_dim),
                self.queue.clone().detach()
            )

            # Insert the positive logit in a random index
            logits = torch.cat([positive_logits.view(-1, 1), negative_logits], dim=1)
            labels = torch.zeros(batch_size, dtype=torch.long).to(logits.device)

        else:
            # Linear predictor
            logits = self.linear_mlp(q)
            labels = []

        # Apply temperature
        predictions = (logits / self.temperature).type(torch.double)

        outputs = {
            Predictions: predictions,
            Labels: labels,
        }

        return outputs

    def __call__(self, inputs: dict) -> Dict[str, Tensor]:
        return self.forward(inputs=inputs)
