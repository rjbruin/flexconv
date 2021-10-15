import torch


def rel_positions_grid(grid_sizes):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int
    """
    tensors = []
    for size in grid_sizes:
        tensors.append(torch.linspace(-1, 1, steps=size))
    # tensors = tuple(dim * [torch.linspace(-1, 1, steps=grid_length)])
    relpos_grid = torch.stack(torch.meshgrid(*tensors), dim=-0)
    return relpos_grid
