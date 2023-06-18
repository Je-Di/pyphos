import numpy.typing as npt
import torch

from .source import Source


class Laser(Source):
    """Generates a tensor with optical power data."""

    def __init__(self, output_powers: npt.NDArray) -> None:
        super().__init__()
        self._base_output = torch.tensor(output_powers)

    def generate_output(
        self, batch_size: int, sequence_length: int, requires_grad: bool = False
    ) -> torch.Tensor:
        """Generate tensor with optical power data.

        The output is of shape [batch_size][sequence_length][port_count][wavelength_count].

        Arguments:
            batch_size -- Number of samples in the batch

        Returns:
            Tensor with optical power data
        """

        # Add batch and sequence dimensions to the base output and expand them to the required size.
        output = self._base_output.unsqueeze(0)
        output = output.unsqueeze(0)
        output = output.expand(batch_size, sequence_length, *output.shape[2:])
        return output.requires_grad_(requires_grad)
