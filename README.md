# PyPhos
Simulator for photonic circuits based on PyTorch.

Optical interface:
[batch_size][sequence_len][port_count][wavelength_count] = power [float]

Analog interface:
[batch_size][sequence_len][port_count] = voltage [float]

Digital interface:
[batch_size][sequence_len][port_count] = value [int]