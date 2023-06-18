"""
Uses the MatrixMultiplier module to calculate convolutions on MNIST images.
"""
import io
import pkgutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pyphos.modules import Eam, MatrixMultiplier
from pyphos.sources import Laser


def image_to_convolution_sequence(
    image_data: torch.Tensor,
    kernel_size: int = 3,
    pad_value: float = 0.5,
):
    """Convert 'image_data' of shape [N][H][W] into sequence of shape [N][S][K^2].

    Arguments:
        image_data -- Batch pixel data of shape [N][H][W] in range [0, 1]

    Keyword Arguments:
        kernel_size -- Edge length of (quadratic) kernel (default: {3})
        pad_value -- Pad image data with this value (default: {0.5})

    Returns:
        Tensor of shape [N][S][K^2].

    To calculate a convolution, all pixel data corresponding to a kernel position must be presented
    to the tensor core simultaneously. The length of the sequence equals the number of pixels of the
    input image while the last dimension equals the number of kernel pixels.
    """

    image_data = image_data.clamp(0, 1)

    batch_size = image_data.size(0)
    height, width = image_data.shape[-2:]
    sequence_length = width * height

    # Pad at the left and top edges to be consistent with current FPGA implementation.
    # In the first sequence, the lower right pixel of the kernel data will be the upper left pixel
    # of the image. The kernel will then move row-wise to the bottom right of the image.
    pad = nn.ConstantPad2d((kernel_size - 1, 0, kernel_size - 1, 0), pad_value)
    image_data = pad(image_data)

    # Create all row and column index combinations and convert them to the flattened image space.
    row_idx = torch.stack([torch.arange(row, row + kernel_size) for row in range(height)])
    col_idx = torch.stack([torch.arange(col, col + kernel_size) for col in range(width)])
    row_idx = row_idx.repeat_interleave(kernel_size, dim=1)
    row_idx = row_idx.repeat_interleave(width, dim=0)
    col_idx = col_idx.repeat(height, kernel_size)
    idx = col_idx + row_idx * (width + kernel_size - 1)
    # Add batch dimension and expand it to the batch size.
    idx.unsqueeze_(0)
    idx = idx.expand(batch_size, *idx.shape[1:])

    # To prepare for the gather operation:
    #   - Add a new dimension before the image dimensions and expand it to the sequence length.
    #   - Flatten the pixels to the same linear space as the indices.
    image_data.unsqueeze_(-3)
    image_data = image_data.expand(batch_size, sequence_length, *image_data.shape[-2:])
    image_data = image_data.view(batch_size, sequence_length, -1)
    sequence = image_data.gather(2, idx)

    return sequence


def get_data_loaders(
    path: str = "files", batch_size: int = 1, test_batch_size: int = 1
) -> tuple[DataLoader]:
    """Adapted from hxtorch.

    Keyword Arguments:
        path -- Path to data (default: {"files"})
        batch_size -- Batch size for the train data (default: {1})
        test_batch_size -- Batch size for the test data (default: {1})

    Returns:
        Tuple of DataLoaders, one for the train and one for the test data
    """
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.CenterCrop((24, 24)),
            transforms.Resize(16, interpolation=PIL.Image.BICUBIC),
        ]
    )
    # Get data
    data_path = Path(path).resolve()
    train_data = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    # Data loaders
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=test_batch_size)
    return train_loader, test_loader


def main():
    """Instantiate photonic tensor core with kernel weights and apply MNIST image data."""
    # The number of parallel input waveguides.
    INPUT_SIZE = 9
    # List of wavelengths used by the modules in nm.
    WAVELENGTHS = [1550]
    # The (unmodulated) input power per input and wavelength in mW.
    POWER = np.ones((INPUT_SIZE, len(WAVELENGTHS)))
    # Matrix weights are defined via these kernels.
    KERNELS = (
        ("Reference", [0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ("Identity", [0, 0, 0, 0, 1, 0, 0, 0, 0]),
        ("Edge Top", [-1, -1, -1, 0, 0, 0, 1, 1, 1]),
        ("Edge Left", [-1, 0, 1, -1, 0, 1, -1, 0, 1]),
        ("Ridge A", [0, -1, 0, -1, 4, -1, 0, -1, 0]),
        ("Ridge B", [-1, -1, -1, -1, 8, -1, -1, -1, -1]),
        (
            "Gaussian Blur",
            [1 / 16, 2 / 16, 1 / 16, 2 / 16, 4 / 16, 2 / 16, 1 / 16, 2 / 16, 1 / 16],
        ),
    )
    EAM_BIAS = -2  # The EAM bias voltage in V.
    # PD_BIAS = -3  # The photodiode bias voltage in V.
    # The modulation range for the input data in Volt.
    # [0, 1] gets mapped to [-V/2, V/2].
    MODULATION_RANGE_INPUT = 1
    # The modulation factor for the kernel data in V.
    MODULATION_FACTOR_KERNEL = 0.2

    output_size = len(KERNELS)
    weights = np.array([kernel[1] for kernel in KERNELS]) * MODULATION_FACTOR_KERNEL + EAM_BIAS
    eam_characterization = io.BytesIO(pkgutil.get_data("pyphos", "data/eam_3mw_1530nm_1570nm.npy"))
    eam_characterization = np.load(eam_characterization)
    eam = Eam(WAVELENGTHS, (1530, 1570), (-3, 0), eam_characterization)
    matrix_multiplier = MatrixMultiplier(WAVELENGTHS, INPUT_SIZE, output_size, weights, eam, eam)

    train_loader, _ = get_data_loaders(batch_size=5, test_batch_size=5)
    image_loader = enumerate(train_loader)
    _, (image_data, _) = next(image_loader)
    # Remove channel dimension.
    image_data.squeeze_()
    convolution_sequence = (
        image_to_convolution_sequence(image_data) - 0.5
    ) * MODULATION_RANGE_INPUT + EAM_BIAS

    HEIGHT, WIDTH = image_data[0].shape
    BATCH_SIZE, SEQUENCE_LENGTH = convolution_sequence.shape[:2]

    laser = Laser(POWER)
    laser_output = laser.generate_output(BATCH_SIZE, SEQUENCE_LENGTH)
    convolution = matrix_multiplier(laser_output, convolution_sequence)
    # Convert sequence data back to 2-D images.
    convolution = convolution.view(BATCH_SIZE, HEIGHT, WIDTH, output_size)
    # Subtract unmodulated reference.
    convolution = convolution - convolution[:, :, :, 0:1]

    _, axs = plt.subplots(1, output_size, figsize=(10, 2))
    axs[0].imshow(image_data[0], cmap="gray", interpolation="none")
    axs[0].set_title("Original")
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    for kernel in range(1, output_size):
        axs[kernel].imshow(convolution[0, 0:, 0:, kernel], cmap="gray", interpolation="none")
        axs[kernel].set_title(KERNELS[kernel][0])
        axs[kernel].set_xticks([])
        axs[kernel].set_yticks([])
    plt.show()


if __name__ == "__main__":
    main()
