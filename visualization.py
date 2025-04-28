import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from physics import source

FIGURES_DIR = os.path.join("rapport", "figures")


# Utility functions for figure saving
def save_figure(filename, tight=True):
    """Save figure to figures directory with appropriate formatting."""
    filepath = os.path.join(FIGURES_DIR, filename)
    if tight:
        plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight", dpi=300)
    plt.close()
    return filename


def plot_solution_2d(X, Y, u, title="Solution", filename=None):
    """Plot 2D contour of the solution."""
    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, u, levels=50, cmap="jet")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    # plt.show()
    if filename:
        return save_figure(filename)


def plot_solution_3d(X, Y, u, title="Solution", filename=None):
    """Plot 3D surface of the solution."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, u, cmap="jet")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    ax.set_title(title)
    plt.tight_layout()
    # plt.show()
    if filename:
        return save_figure(filename)


def plot_error_2d(X, Y, error, title="Error", filename=None):
    """Plot 2D contour of the error."""
    plt.figure(figsize=(6, 5))
    plt.contourf(X, Y, error, levels=50, cmap="coolwarm")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    # plt.show()
    if filename:
        return save_figure(filename)


def plot_error_3d(X, Y, error, title="Error", filename=None):
    """Plot 3D surface of the error."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, error, cmap="coolwarm")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("Error")
    ax.set_title(title)
    plt.tight_layout()
    # plt.show()
    if filename:
        return save_figure(filename)


def plot_training_history(train_losses, val_losses=None, filename=None):
    """Plot training history."""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    if val_losses is not None:
        plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    # plt.show()
    if filename:
        return save_figure(filename)


def calculate_errors(u_pred, u_ref):
    """Calculate error metrics between predicted and reference solutions."""
    error = u_pred - u_ref
    mse = np.mean(error**2)

    # Avoid division by zero
    norm_ref = np.linalg.norm(u_ref)
    if norm_ref < 1e-10:
        rel_l2_error = np.linalg.norm(error)
    else:
        rel_l2_error = np.linalg.norm(error) / norm_ref

    return error, mse, rel_l2_error


def calculate_residuals(X, Y, U):
    """
    Calculate the residual Δu + f(x, y) for the Poisson equation.
    Assumes U is a 2D grid of solution values and X, Y contain the grid points.
    """
    dx = X[0, 1] - X[0, 0]  # Uniform grid spacing in x
    dy = Y[1, 0] - Y[0, 0]  # Uniform grid spacing in y

    # Compute second derivatives using finite differences
    d2u_dx2 = (U[2:, 1:-1] - 2 * U[1:-1, 1:-1] + U[:-2, 1:-1]) / dx**2
    d2u_dy2 = (U[1:-1, 2:] - 2 * U[1:-1, 1:-1] + U[1:-1, :-2]) / dy**2

    # Laplacian
    delta_u = d2u_dx2 + d2u_dy2

    # Combine X and Y into a single array of points for the source function
    x_flat = X[1:-1, 1:-1].flatten()
    y_flat = Y[1:-1, 1:-1].flatten()
    xy_points = np.stack([x_flat, y_flat], axis=-1)  # Shape: (num_points, 2)

    # Source term
    f = source(xy_points).reshape(X[1:-1, 1:-1].shape)  # Reshape to match grid

    # Residual
    residu = delta_u + f
    return residu


def print_error_metrics(mse, rel_l2_error, method1="Pred", method2="Ref"):
    """Print error metrics in a formatted way."""
    print(f"Mean Squared Error ({method1} vs {method2}): {mse:.6e}")
    print(f"Relative L2 Error ({method1} vs {method2}): {rel_l2_error:.6f}")
