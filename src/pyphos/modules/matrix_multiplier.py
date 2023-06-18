"""
Build a photonic tensor core (https://www.nature.com/articles/s41586-020-03070-1).
"""
import numpy as np
import numpy.typing as npt
import torch

from ..utilities import db_to_factor
from .modulator import Modulator
from .phos_module import PhosModule
from .photodiode import Photodiode
from .splitter import Splitter


class MatrixMultiplier(PhosModule):
    """A matrix vector multiplication structure."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        wavelengths: npt.NDArray,
        input_size: int,
        output_size: int,
        weights: npt.NDArray,
        input_modulator: Modulator,
        weight_modulator: Modulator,
        input_loss_db: float = 3.0,
    ) -> None:
        """
        Arguments:
            input_size -- The number of optical input ports
            output_size -- The number of photodiode outputs
            weights -- The (initial) voltages applied to the weight modulators

        Raises:
            ValueError: When the size of 'weights' does not agree with the port sizes
        """
        super().__init__(wavelengths)
        self.input_size = input_size
        self.output_size = output_size
        if weights.shape != (output_size, input_size):
            raise ValueError(
                f"Size of weight matrix {weights.shape} does not agree with "
                f"matrix size {input_size, output_size}."
            )
        self.weights = torch.tensor(weights.flatten()).view(1, 1, -1)
        self.input_loss_db = input_loss_db
        self.input_modulator = input_modulator
        self.weight_modulator = weight_modulator
        self.kernel_split = Splitter(wavelengths, np.ones(output_size) / output_size)
        self.photodiode = Photodiode(wavelengths, 1)

    def forward(self, input_optical: torch.Tensor, modulation_analog: torch.Tensor) -> torch.Tensor:
        """Process optical input data using the photonic tensor core.

        Arguments:
            input_optical -- Optical input data
            modulation_analog -- Analog voltages on the input modulators

        Returns:
            Photodiode currents
        """
        batch_size, sequence_length = input_optical.shape[:2]

        # Apply loss due to input coupling.
        # TODO: Replace with grating coupler module.
        output = input_optical / db_to_factor(self.input_loss_db)
        # Modulate the incoming light.
        output = self.input_modulator(output, modulation_analog)
        # Split into equal parts for every kernel.
        output = self.kernel_split(output)
        # Modulate by the weights.
        output = self.weight_modulator(output, self.weights)
        # Divide by the number of inputs.
        # This corresponds to coupling the modulated light onto the summmation waveguides.
        output /= self.input_size
        # Sum per output waveguide.
        # TODO: Replace with summing module.
        output = output.view(batch_size, sequence_length, self.output_size, self.input_size, -1)
        output = output.sum(3)
        # Convert to current.
        output = self.photodiode(output)
        return output
