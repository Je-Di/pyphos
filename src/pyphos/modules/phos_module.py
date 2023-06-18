from abc import ABCMeta, abstractmethod

import numpy.typing as npt
import torch
from torch import nn


class PhosModule(nn.Module, metaclass=ABCMeta):
    """Abstract base class for all photonic modules."""

    def __init__(self, wavelengths: npt.NDArray):
        super().__init__()
        self.wavelengths = wavelengths

    # pylint: disable=missing-function-docstring
    @abstractmethod
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        pass
