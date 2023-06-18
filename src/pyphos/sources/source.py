from abc import ABCMeta, abstractmethod

import torch


class Source(metaclass=ABCMeta):
    """Abstract base class for optical, analog or digital sources."""

    @abstractmethod
    def generate_output(
        self, batch_size: int, sequence_length: int, requires_grad: bool = False
    ) -> torch.Tensor:
        """Create tensor to be forwarded to photonic modules.

        Arguments:
            batch_size -- Batch size
            sequence_length -- Sequence length

        Keyword Arguments:
            requires_grad -- 'requires_grad' setting of the returned tensor (default: {False})

        Returns:
            Tensor with source data. As this data will typically be the leaf of the module,
            'requires_grad' can be set.
        """
