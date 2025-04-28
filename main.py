#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate a LaTeX report for the Hybrid AI course project.
This script generates figures and LaTeX files in the rapport directory.
"""

import os
import time
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rc
from sklearn.model_selection import train_test_split

# Import project modules
from config import device, params  # Import params dictionary
from data_utils import (
    generate_evaluation_grid,
    generate_grid_points,
    separate_interior_boundary_points,
)
from fd_solver import solve_poisson_fd
from models import PINN, SupervisedNet
from train import train_pinn, train_supervised
from visualization import (
    calculate_errors,
    calculate_residuals,
    plot_error_2d,
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


def run_computations():
    """
    Run the computational methods and save figures for the report.
    Returns a dictionary with results and figure filenames.
    """
    print("Starting computations...")
    # Initialize results with parameters
    records = {"parameters": params.copy()}
    timings = {
        "PINN": {
            "Training": None,
            "Inference": None,
        },
        "Supervised Regression": {"Training": None, "Inference": None},
        "FD": {"Training": None, "Inference": None},
        "Data Generation": None,
    }

    print("Using device:", device)

    # Part 1: PINN (Physics-Informed Neural Network) Training
    print("\n===== Part 1: PINN Training =====")

    # Generate grid points
    start_time = time.time()
    X_grid, Y_grid, XY_grid, x_lin, y_lin = generate_grid_points(
        n_points=params["n_points_per_dim"]
    )
    timings["Data Generation"] = time.time() - start_time

    # Separate interior and boundary points
    x_interior, x_boundary, u_boundary = separate_interior_boundary_points(XY_grid)

    # Create PINN model
    pinn_model = PINN().to(device)

    # Train PINN model

    start_time = time.time()
    pinn_losses = train_pinn(pinn_model, x_interior, x_boundary, u_boundary)
    timings["PINN"]["Training"] = time.time() - start_time

    # Plot PINN training history
    pinn_history_file = plot_training_history(
        pinn_losses, None, "pinn_training_history.pdf"
    )
    records["pinn_history_file"] = pinn_history_file

    # Generate evaluation grid
    X, Y, xy_tensor = generate_evaluation_grid(params["n_plot"])

    # Evaluate PINN on grid
    pinn_model.eval()
    start_time = time.time()
    with torch.no_grad():
        u_pred = (
            pinn_model(xy_tensor)
            .cpu()
            .numpy()
            .reshape(params["n_plot"], params["n_plot"])
        )
    timings["PINN"]["Inference"] = time.time() - start_time

    # Plot PINN solution
    pinn_2d_file = plot_solution_2d(
        X,
        Y,
        u_pred,
        "PINN Solution for Poisson Problem",
        "pinn_solution_2d.pdf",
    )
    pinn_3d_file = plot_solution_3d(
        X,
        Y,
        u_pred,
        "PINN Solution for Poisson Problem",
        "pinn_solution_3d.pdf",
    )
    records["pinn_2d_file"] = pinn_2d_file
    records["pinn_3d_file"] = pinn_3d_file

    residuals = calculate_residuals(X, Y, u_pred)
    residuals_2d = plot_error_2d(
        X[1:-1, 1:-1],
        Y[1:-1, 1:-1],
        residuals,
        title="PINN Solution Residuals",
        filename="pinn_residuals_2d.pdf",
    )
    residuals_3d = plot_error_3d(
        X[1:-1, 1:-1],
        Y[1:-1, 1:-1],
        residuals,
        title="PINN Solution Residuals",
        filename="pinn_residuals_3d.pdf",
    )
    records["pinn_residuals_2d_file"] = residuals_2d
    records["pinn_residuals_3d_file"] = residuals_3d

    # Solve using Finite Differences for reference
    print("\n===== Finite Difference Reference Solution =====")
    start_time = time.time()
    x_fd, y_fd, u_fd = solve_poisson_fd(
        params["n_plot"], params["n_plot"], params["a"], params["b"]
    )
    timings["FD"]["Training"] = (
        time.time() - start_time
    )  # Not really training, but computation time
    timings["FD"]["Inference"] = (
        0.0  # FD solution is directly computed, no separate inference step
    )

    # Plot FD solution
    fd_2d_file = plot_solution_2d(
        X,
        Y,
        u_fd,
        "Finite Difference Solution (Reference)",
        "fd_solution_2d.pdf",
    )
    fd_3d_file = plot_solution_3d(
        X,
        Y,
        u_fd,
        "Finite Difference Solution (Reference)",
        "fd_solution_3d.pdf",
    )
    records["fd_2d_file"] = fd_2d_file
    records["fd_3d_file"] = fd_3d_file

    # Compare PINN vs FD
    print("\n===== Comparing PINN vs FD =====")
    error_pinn_fd, mse_pinn_fd, rel_l2_error_pinn_fd = calculate_errors(u_pred, u_fd)
    records["mse_pinn_fd"] = mse_pinn_fd
    records["rel_l2_error_pinn_fd"] = rel_l2_error_pinn_fd

    # Plot error
    pinn_error_2d_file = plot_error_2d(
        X,
        Y,
        error_pinn_fd,
        "Error (PINN - FD)",
        "pinn_fd_error_2d.pdf",
    )
    pinn_error_3d_file = plot_error_3d(
        X,
        Y,
        error_pinn_fd,
        "Error (PINN - FD)",
        "pinn_fd_error_3d.pdf",
    )
    records["pinn_error_2d_file"] = pinn_error_2d_file
    records["pinn_error_3d_file"] = pinn_error_3d_file

    # Part 2: Supervised Learning using FD solution
    print("\n===== Part 2: Supervised Learning =====")

    # Prepare data from FD solution
    X_fd_flat = np.stack([X.flatten(), Y.flatten()], axis=-1)
    y_fd_flat = u_fd.flatten()

    # Split into train/validation/test sets
    X_train_np, X_temp_np, y_train_np, y_temp_np = train_test_split(
        X_fd_flat, y_fd_flat, test_size=0.2, random_state=42
    )
    X_val_np, X_test_np, y_val_np, y_test_np = train_test_split(
        X_temp_np, y_temp_np, test_size=0.5, random_state=42
    )

    # Convert to tensors
    X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1).to(device)
    X_val = torch.tensor(X_val_np, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val_np, dtype=torch.float32).unsqueeze(1).to(device)
    X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test_np, dtype=torch.float32).unsqueeze(1).to(device)

    # Create and train supervised model
    model_sup = SupervisedNet().to(device)
    start_time = time.time()
    train_losses, val_losses = train_supervised(
        model_sup, X_train, y_train, X_val, y_val
    )
    timings["Supervised Regression"]["Training"] = time.time() - start_time

    # Plot training history
    sup_history_file = plot_training_history(
        train_losses,
        val_losses,
        "supervised_training_history.pdf",
    )
    records["sup_history_file"] = sup_history_file

    # Evaluate supervised model on grid
    model_sup.eval()
    start_time = time.time()
    with torch.no_grad():
        u_pred_sup = (
            model_sup(xy_tensor)
            .cpu()
            .numpy()
            .reshape(params["n_plot"], params["n_plot"])
        )
    timings["Supervised Regression"]["Inference"] = time.time() - start_time

    # Plot supervised solution
    sup_2d_file = plot_solution_2d(
        X,
        Y,
        u_pred_sup,
        "Supervised Regression Solution",
        "supervised_solution_2d.pdf",
    )
    sup_3d_file = plot_solution_3d(
        X,
        Y,
        u_pred_sup,
        "Supervised Regression Solution",
        "supervised_solution_3d.pdf",
    )
    records["sup_2d_file"] = sup_2d_file
    records["sup_3d_file"] = sup_3d_file

    residuals_sup = calculate_residuals(X, Y, u_pred_sup)
    residuals_sup_2d = plot_error_2d(
        X[1:-1, 1:-1],
        Y[1:-1, 1:-1],
        residuals_sup,
        title="Supervised Regression Solution Residuals",
        filename="sup_residuals_2d.pdf",
    )
    residuals_sup_3d = plot_error_3d(
        X[1:-1, 1:-1],
        Y[1:-1, 1:-1],
        residuals_sup,
        title="Supervised Regression Solution Residuals",
        filename="sup_residuals_3d.pdf",
    )
    records["sup_residuals_2d_file"] = residuals_sup_2d
    records["sup_residuals_3d_file"] = residuals_sup_3d

    # Compare Supervised vs FD
    print("\n===== Comparing Supervised vs FD =====")
    error_sup_fd, mse_sup_fd, rel_l2_error_sup_fd = calculate_errors(u_pred_sup, u_fd)
    records["mse_sup_fd"] = mse_sup_fd
    records["rel_l2_error_sup_fd"] = rel_l2_error_sup_fd

    # Plot error
    sup_error_2d_file = plot_error_2d(
        X,
        Y,
        error_sup_fd,
        "Error (Supervised - FD)",
        "supervised_fd_error_2d.pdf",
    )
    sup_error_3d_file = plot_error_3d(
        X,
        Y,
        error_sup_fd,
        "Error (Supervised - FD)",
        "supervised_fd_error_3d.pdf",
    )
    records["sup_error_2d_file"] = sup_error_2d_file
    records["sup_error_3d_file"] = sup_error_3d_file

    # Evaluate on test set
    model_sup.eval()
    with torch.no_grad():
        y_pred_test = model_sup(X_test).cpu().numpy()
        y_test_np = y_test.cpu().numpy()

    # Calculate test errors
    _, mse_sup_test, rel_l2_error_sup_test = calculate_errors(y_pred_test, y_test_np)
    records["mse_sup_test"] = mse_sup_test
    records["rel_l2_error_sup_test"] = rel_l2_error_sup_test

    print("\n===== Comparison of Errors vs FD Reference =====")
    print(
        f"  PINN (on full grid): MSE: {mse_pinn_fd:.6e}, Relative L2: {rel_l2_error_pinn_fd:.6f}"
    )
    print(
        f"  Supervised (on full grid): MSE: {mse_sup_fd:.6e}, Relative L2: {rel_l2_error_sup_fd:.6f}"
    )
    print(
        f"  Supervised (on test set): MSE: {mse_sup_test:.6e}, Relative L2: {rel_l2_error_sup_test:.6f}"
    )

    # Add timings to results
    records["timings"] = timings

    return records


def main():
    """Main function to run computations and generate the report."""
    # Run computations and generate figures
    records = run_computations()

    print("=== Records ===")
    pprint(records)

    # Save records as json
    records_file = os.path.join(REPORT_DIR, "records.json")
    with open(records_file, "w") as f:
        import json

        def convert_to_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.tolist()  # Convert tensors to lists
            if isinstance(obj, np.ndarray):
                return obj.tolist()  # Convert numpy arrays to lists
            if isinstance(obj, np.float32):
                return float(obj)  # Convert numpy float32 to Python float
            return obj

        json.dump(records, f, indent=4, default=convert_to_serializable)
    print(f"Records saved to: {records_file}")

    print("\nReport generation complete!")
    print(f"  - Figures saved to: {FIGURES_DIR}")
    print(f"  - LaTeX files saved to: {REPORT_DIR}")
    print("\nTo compile the report, navigate to the rapport directory and run:")
    print("  pdflatex rapport.tex")


if __name__ == "__main__":
    main()
