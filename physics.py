import numpy as np
import torch

from config import params, pi


def source(x):
    """
    Source term for the Poisson equation.
    f(x,y) = x*sin(a*pi*y) + y*sin(b*pi*x)
    """
    # Ensure x is a tensor for source calculation if needed elsewhere
    if isinstance(x, np.ndarray):
        x_tensor = torch.tensor(x, dtype=torch.float32)
    else:
        x_tensor = x

    f_val = x_tensor[:, 1:2] * torch.sin(
        params["a"] * pi * x_tensor[:, 0:1]
    ) + x_tensor[:, 0:1] * torch.sin(params["b"] * pi * x_tensor[:, 1:2])

    # Return numpy if input was numpy, for FD solver
    if isinstance(x, np.ndarray):
        return f_val.numpy()
    else:
        return f_val


def poisson_residual(model, x):
    """
    Calculate the residual of the Poisson equation.
    Δu + f(x,y) = 0
    """
    u = model(x)

    # First derivatives
    grad_u = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True
    )[0]

    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]

    # Second derivatives
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True
    )[0][:, 0:1]

    u_yy = torch.autograd.grad(
        u_y, x, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True
    )[0][:, 1:2]

    # Laplacian
    laplacian = u_xx + u_yy

    # Residual: Δu + f = 0
    return laplacian + source(x)
