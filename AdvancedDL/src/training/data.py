from typing import Tuple
from torch import Tensor
from PIL import ImageFilter

import numpy as np


class RandomCropsTransform(object):
    def __init__(self, base_transform):
        super(RandomCropsTransform, self).__init__()
        self.base_transform = base_transform

    def __call__(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        query = self.base_transform(x)
        key = self.base_transform(x)

        return query, key


class GaussianBlur(object):
    def __init__(self, dist_params: Tuple[float, float] = (0.1, 2.0)):
        self.dist_params = dist_params

    def __call__(self, x: Tensor) -> Tensor:
        sigma = np.random.uniform(
            low=self.dist_params[0],
            high=self.dist_params[1],
            size=(1,),
        )
        x = x.filter(
            ImageFilter.GaussianBlur(radius=sigma),
        )

        return x
