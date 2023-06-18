from abc import ABCMeta

import numpy.typing as npt
import torch
import torch.nn.functional as F

from .phos_module import PhosModule


class Modulator(PhosModule, metaclass=ABCMeta):
    """Modulates the optical input according to the analog modulation voltage.

    TODO: Currently, this class does not add any functionality to the base class and is used for
    type hinting only. Remove?

    def forward(self, input_optical: torch.Tensor, modulation_analog: torch.Tensor) -> torch.Tensor:

    Any of the three modulation_analog dimensions may have size 1 in which case they are broadcasted
    to the size of corresponding input dimension.
    """


class LinearModulator(Modulator):
    """Linear modulation according to mod = m * v + b."""

    # pylint: disable=invalid-name
    def __init__(self, wavelengths: npt.NDArray, m: float, b: float) -> None:
        super().__init__(wavelengths)
        self.m = m
        self.b = b

    # pylint: disable=arguments-differ
    def forward(self, input_optical: torch.Tensor, modulation_analog: torch.Tensor) -> torch.Tensor:
        modulation_factors = modulation_analog * self.m + self.b
        # As the modulation is independent of the wavelength, add a dimension at the end to enable
        # broadcasting.
        modulation_factors = modulation_factors.unsqueeze(-1)
        return input_optical * modulation_factors


class Eam(Modulator):
    """Modulation according to EAM characteristic.

    # TODO: Make more general as this type of modulator can be used for any modulator with measured
    or calculated transfer function that depends on voltage and wavelength.
    """

    def __init__(
        self,
        wavelengths: npt.NDArray,
        characterization_wavelengths: tuple[float, float],
        characterization_voltages: tuple[float, float],
        characterization_data: npt.NDArray,
    ) -> None:
        super().__init__(wavelengths)
        self.characterization_wavelengths = characterization_wavelengths
        self.characterization_voltages = characterization_voltages
        self.characterization_data = torch.tensor(characterization_data)
        # Add batch and channel dimensions to prepare for 'grid_sample' function.
        self.characterization_data = self.characterization_data.unsqueeze(0).unsqueeze(0)
        # A coordinate grid is required to sample from the characterization data.
        # As the modulation factor depends on the voltage and wavelength a '_grid_base' with fixed
        # fixed wavelengths is created here.
        self._grid_base = (torch.tensor(self.wavelengths) - characterization_wavelengths[0]) / (
            characterization_wavelengths[1] - characterization_wavelengths[0]
        ) * 2 - 1
        # 'torch.functional.grid_sample()' requires 'double' data type.
        self._grid_base = self._grid_base.to(torch.double)

    # pylint: disable=arguments-differ
    def forward(self, input_optical: torch.Tensor, modulation_analog: torch.Tensor) -> torch.Tensor:
        batch_size_mod, sequence_length_mod, port_count_mod = modulation_analog.shape
        # The modulation factors depend on the wavelength and modulation voltage.
        # Create a grid with all the relative coordinates of the characterization data to be used
        # with 'F.grid_sample()'.
        # As we have 2-D characterization data, the grid must have shape [N][Hout][Wout][2].
        # The grid's shape is [N][sequence_len * port_count][wavelength_count][2].
        # The wavelengths are created from '_grid_base', which has already been scaled to [-1, 1].
        # The modulation voltages are the same for all wavelengths.
        modulation_analog = modulation_analog.flatten()
        grid_wavelengths = self._grid_base.expand(modulation_analog.size(0), -1).flatten()
        # Scale the modulation voltages to [-1, 1].
        grid_voltages = (modulation_analog - self.characterization_voltages[0]) / (
            self.characterization_voltages[1] - self.characterization_voltages[0]
        ) * 2 - 1
        # Interleaved repeat without copying data.
        grid_voltages = grid_voltages.unsqueeze(-1)
        grid_voltages = grid_voltages.expand(-1, len(self.wavelengths)).flatten()
        # Stack the voltages and wavelengths and view according to original size.
        grid = torch.stack((grid_wavelengths, grid_voltages), 1)
        grid = grid.view(
            batch_size_mod, sequence_length_mod * port_count_mod, len(self.wavelengths), 2
        )
        # Now the modulation factors can be interpolated from the characterization data. As the
        # boundaries of the characterization data correspond to the edges of the intervals,
        # 'align_corners' is set to True.
        modulation_factors = F.grid_sample(
            self.characterization_data.expand(batch_size_mod, -1, -1, -1), grid, align_corners=True
        )
        # Resize to original size, preparing for broadcast element-wise multiplication.
        modulation_factors = modulation_factors.view(
            batch_size_mod, sequence_length_mod, port_count_mod, len(self.wavelengths)
        )
        return input_optical * modulation_factors
