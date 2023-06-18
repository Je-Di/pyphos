import warnings

import numpy.typing as npt
import torch

from .phos_module import PhosModule


class Splitter(PhosModule):
    """Splits optical input into optical output with increased port size."""

    def __init__(self, wavelengths: npt.NDArray, splitting_ratios: npt.NDArray) -> None:
        super().__init__(wavelengths)
        if sum(splitting_ratios) > 1:
            warnings.warn(f"Summed splitting ratios exceed one ({sum(splitting_ratios):.2f}).")
        self.splitting_ratios = torch.tensor(splitting_ratios)
        self.output_count = len(splitting_ratios)

    # pylint: disable=arguments-differ,redefined-builtin
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """(P0, P1, P2, ...) -> (P0_0, P0_1, ..., P1_0, P1_1, ..., P2_0, P2_1, ...)

        Arguments:
            input -- Optical input

        Returns:
            Optical output with increased port size

        The port size of the output data is increased by a factor equal to the number of splitting
        ratios.
        """
        # Insert dimension before port dimension.
        output = input.unsqueeze(-3).tile(self.output_count, 1, 1)
        # Multiply splitting ratios along this new dimension.
        output *= self.splitting_ratios.view(self.output_count, 1, 1)
        # Flatten into single port dimension.
        return output.view(*input.shape[:2], -1, input.size(-1))
