import numpy as np
import torch

# --- General & Physics Parameters ---
params = {
    # General
    "n_points_per_dim": 100,  # Grid resolution for point generation
    "n_plot": 50,  # Grid resolution for evaluation and plotting
    # Physics (Source Term)
    "a": 0.3,
    "b": 0.5,
    # PINN Training
    "pinn_lr": 1e-3,
    "pinn_epochs": 10_000,
    # Supervised Training
    "sup_lr": 1e-3,
    "sup_epochs": 10_000,
    "sup_batch_size": 128,
}

# --- Constants ---
pi = np.pi

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
