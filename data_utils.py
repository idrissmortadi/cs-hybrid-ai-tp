import numpy as np
import torch

from config import device, params


def generate_grid_points(n_points=params["n_points_per_dim"]):
    """Generate a grid of points in [0,1]x[0,1] domain."""
    # Generate grid points
    x_lin = np.linspace(0, 1, n_points)
    y_lin = np.linspace(0, 1, n_points)
    X_grid, Y_grid = np.meshgrid(x_lin, y_lin)
    XY_grid = np.stack([X_grid.flatten(), Y_grid.flatten()], axis=-1)

    return X_grid, Y_grid, XY_grid, x_lin, y_lin


def separate_interior_boundary_points(XY_grid):
    """Separate interior and boundary points from a grid."""
    # Identify boundary points (points on the edges of the unit square)
    is_boundary = (
        (XY_grid[:, 0] == 0.0)
        | (XY_grid[:, 0] == 1.0)
        | (XY_grid[:, 1] == 0.0)
        | (XY_grid[:, 1] == 1.0)
    )

    # Separate interior and boundary points
    X_interior = XY_grid[~is_boundary]
    X_boundary = XY_grid[is_boundary]

    # Count points
    n_interior = X_interior.shape[0]
    n_boundary = X_boundary.shape[0]

    print(f"Generated {n_interior} interior points and {n_boundary} boundary points.")

    # Convert to tensors and move to device
    x_interior = torch.tensor(X_interior, dtype=torch.float32, requires_grad=True).to(
        device
    )
    x_boundary = torch.tensor(X_boundary, dtype=torch.float32).to(device)
    u_boundary = torch.zeros((n_boundary, 1), dtype=torch.float32).to(device)

    return x_interior, x_boundary, u_boundary


def generate_evaluation_grid(n_points):
    """Generate a grid for evaluation and plotting."""
    x = np.linspace(0, 1, n_points)
    y = np.linspace(0, 1, n_points)
    X, Y = np.meshgrid(x, y)
    xy = np.stack([X.flatten(), Y.flatten()], axis=-1)
    xy_tensor = torch.tensor(xy, dtype=torch.float32).to(device)

    return X, Y, xy_tensor
