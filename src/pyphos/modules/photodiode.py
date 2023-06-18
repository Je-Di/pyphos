import numpy.typing as npt
import torch

from .phos_module import PhosModule


class Photodiode(PhosModule):
    """Translates optical input into analog output.

    # TODO: Implement wavelength dependency.
    """

    def __init__(self, wavelengths: npt.NDArray, gains: npt.NDArray) -> None:
        super().__init__(wavelengths)
        self.gains = torch.tensor(gains)

    # pylint: disable=arguments-differ,redefined-builtin
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (input * self.gains).sum(-1)
