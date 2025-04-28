import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from config import params
from physics import source


def solve_poisson_fd(nx, ny, a=params["a"], b=params["b"]):
    """
    Solve Poisson equation using Finite Differences on [0,1]x[0,1] with u=0 BC.
    Δu + f(x,y) = 0 with u=0 on the boundary
    """
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dx2 = dx**2
    dy2 = dy**2

    # Grid for source term evaluation
    X, Y = np.meshgrid(x, y)
    xy_grid = np.stack([X.flatten(), Y.flatten()], axis=-1)

    # Evaluate source term f on the grid
    f_grid = source(xy_grid).reshape(ny, nx)

    # Number of interior points
    nx_int = nx - 2
    ny_int = ny - 2
    N = nx_int * ny_int

    # Assemble the matrix A (discrete Laplacian)
    main_diag = np.ones(N) * (-2 / dx2 - 2 / dy2)
    off_diag_x = np.ones(N - 1) / dx2
    off_diag_y = np.ones(N - nx_int) / dy2

    # Adjust off_diag_x for boundaries between rows
    for i in range(1, ny_int):
        off_diag_x[i * nx_int - 1] = 0

    diagonals = [main_diag, off_diag_x, off_diag_x, off_diag_y, off_diag_y]
    offsets = [0, -1, 1, -nx_int, nx_int]
    A = sp.diags(diagonals, offsets, shape=(N, N), format="csr")

    # Assemble the right-hand side vector F
    # We need -f evaluated at interior points
    F = -f_grid[1:-1, 1:-1].flatten()

    # Solve the linear system AU = F
    U_int = spsolve(A, F)

    # Reshape solution and add boundary conditions (u=0)
    u_sol = np.zeros((ny, nx))
    u_sol[1:-1, 1:-1] = U_int.reshape((ny_int, nx_int))

    return x, y, u_sol
