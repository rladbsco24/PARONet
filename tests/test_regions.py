import torch
import importlib

# Import the get_region_mask function without triggering main execution
module = importlib.import_module('main')
get_region_mask = module.get_region_mask


def test_get_region_mask_basic():
    coords = torch.tensor([
        [0.0, 0.0, 1.0],
        [0.4, 0.0, 1.0],
        [0.95, 0.0, 1.0],
        [-0.95, 0.0, 1.0],
        [1.05, 0.0, 1.0],
        [0.0, 0.0, -0.05],
        [0.0, 0.0, 2.05],
        [1.5, 0.0, 1.0],
    ], dtype=torch.float32)
    mask = get_region_mask(
        coords,
        target_region_center=(0.0, 0.0, 1.0),
        target_radius=1.0,
        boundary_range=0.1,
        xy_range=1.0,
        z_range=(0.0, 2.0),
    )
    expected = torch.tensor([1, 1, 1, 1, 2, 2, 2, 0])
    assert torch.equal(mask.cpu(), expected)
