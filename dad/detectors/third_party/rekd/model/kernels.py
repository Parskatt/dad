import math
import torch


def gaussian_multiple_channels(num_channels, sigma):
    r = 2 * sigma
    size = 2 * r + 1
    size = int(math.ceil(size))
    x = torch.arange(0, size, 1, dtype=torch.float)
    y = x.unsqueeze(1)
    x0 = y0 = r

    gaussian = torch.exp(-1 * (((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigma**2)))) / (
        (2 * math.pi * (sigma**2)) ** 0.5
    )
    gaussian = gaussian.to(dtype=torch.float32)

    weights = torch.zeros((num_channels, num_channels, size, size), dtype=torch.float32)
    for i in range(num_channels):
        weights[i, i, :, :] = gaussian

    return weights


def ones_multiple_channels(size, num_channels):
    ones = torch.ones((size, size))
    weights = torch.zeros((num_channels, num_channels, size, size), dtype=torch.float32)

    for i in range(num_channels):
        weights[i, i, :, :] = ones

    return weights


def grid_indexes(size):
    weights = torch.zeros((2, 1, size, size), dtype=torch.float32)

    columns = []
    for idx in range(1, 1 + size):
        columns.append(torch.ones((size)) * idx)
    columns = torch.stack(columns)

    rows = []
    for idx in range(1, 1 + size):
        rows.append(torch.tensor(range(1, 1 + size)))
    rows = torch.stack(rows)

    weights[0, 0, :, :] = columns
    weights[1, 0, :, :] = rows

    return weights


def get_kernel_size(factor):
    """
    Find the kernel size given the desired factor of upsampling.
    """
    return 2 * factor - factor % 2


def linear_upsample_weights(half_factor, number_of_classes):
    """
    Create weights matrix for transposed convolution with linear filter
    initialization.
    """

    filter_size = get_kernel_size(half_factor)

    weights = torch.zeros(
        (
            number_of_classes,
            number_of_classes,
            filter_size,
            filter_size,
        ),
        dtype=torch.float32,
    )

    upsample_kernel = torch.ones((filter_size, filter_size))
    for i in range(number_of_classes):
        weights[i, i, :, :] = upsample_kernel

    return weights


class Kernels_custom:
    def __init__(self, args, MSIP_sizes=[]):
        self.batch_size = args.batch_size
        # create_kernels
        self.kernels = {}

        if MSIP_sizes != []:
            self.create_kernels(MSIP_sizes)

        if 8 not in MSIP_sizes:
            self.create_kernels([8])

    def create_kernels(self, MSIP_sizes):
        # Grid Indexes for MSIP
        for ksize in MSIP_sizes:
            ones_kernel = ones_multiple_channels(ksize, 1)
            indexes_kernel = grid_indexes(ksize)
            upsample_filter_np = linear_upsample_weights(int(ksize / 2), 1)

            self.ones_kernel = ones_kernel.requires_grad_(False)
            self.kernels["ones_kernel_" + str(ksize)] = self.ones_kernel

            self.upsample_filter_np = upsample_filter_np.requires_grad_(False)
            self.kernels["upsample_filter_np_" + str(ksize)] = self.upsample_filter_np

            self.indexes_kernel = indexes_kernel.requires_grad_(False)
            self.kernels["indexes_kernel_" + str(ksize)] = self.indexes_kernel

    def get_kernels(self, device):
        kernels = {}
        for k, v in self.kernels.items():
            kernels[k] = v.to(device)
        return kernels
