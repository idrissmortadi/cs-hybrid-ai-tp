import os

import matplotlib.pyplot as plt
import torch
from matplotlib import rc

# Import project modules
from config import device, params  # Import params dictionary
from data_utils import (
    generate_evaluation_grid,
    generate_grid_points,
    separate_interior_boundary_points,
)
from models import PINN
from train import train_pinn
from visualization import (
    calculate_residuals,
    plot_error_3d,
    plot_solution_2d,
    plot_solution_3d,
    plot_training_history,
)

# Set matplotlib settings for LaTeX output
plt.style.use("seaborn-v0_8-whitegrid")
rc("text", usetex=True)
rc("font", family="serif", size=12)
plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["figure.dpi"] = 300

# Create directories for report and figures
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rapport")
FIGURES_DIR = os.path.join(REPORT_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

COEFFS = [4, 8, 16]

for COEFF in COEFFS:
    print("Starting computations...")
    # Initialize results with parameters

    print("Using device:", device)

    # Part 1: PINN (Physics-Informed Neural Network) Training
    print("\n===== Part 1: PINN Training =====")

    # Generate grid points
    X_grid, Y_grid, XY_grid, x_lin, y_lin = generate_grid_points(
        n_points=params["n_points_per_dim"]
    )

    # Separate interior and boundary points
    x_interior, x_boundary, u_boundary = separate_interior_boundary_points(XY_grid)

    # Create PINN model
    pinn_model = PINN().to(device)

    # Train PINN model

    pinn_losses = train_pinn(
        pinn_model, x_interior, x_boundary, u_boundary, boundary_coefficient=COEFF
    )

    # Plot PINN training history
    pinn_history_file = plot_training_history(
        pinn_losses, None, f"coeff_{COEFF}_pinn_training_history.pdf"
    )

    # Generate evaluation grid
    X, Y, xy_tensor = generate_evaluation_grid(params["n_plot"])

    # Evaluate PINN on grid
    pinn_model.eval()
    with torch.no_grad():
        u_pred = (
            pinn_model(xy_tensor)
            .cpu()
            .numpy()
            .reshape(params["n_plot"], params["n_plot"])
        )

    # Plot PINN solution
    pinn_2d_file = plot_solution_2d(
        X,
        Y,
        u_pred,
        "PINN Solution for Poisson Problem",
        f"coeff_{COEFF}_pinn_solution_2d.pdf",
    )
    pinn_3d_file = plot_solution_3d(
        X,
        Y,
        u_pred,
        "PINN Solution for Poisson Problem",
        f"coeff_{COEFF}_pinn_solution_3d.pdf",
    )
    residuals = calculate_residuals(X, Y, u_pred)
    residuals_3d = plot_error_3d(
        X[1:-1, 1:-1],
        Y[1:-1, 1:-1],
        residuals,
        title="PINN Solution Residuals",
        filename=f"coeff_{COEFF}_pinn_residuals_3d.pdf",
    )
