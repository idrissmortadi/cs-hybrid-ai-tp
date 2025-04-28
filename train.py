import torch
import torch.nn as nn
import torch.optim as optim

from config import device, params
from physics import poisson_residual


def train_pinn(model, x_interior, x_boundary, u_boundary, boundary_coefficient=1.0):
    """
    Train a PINN model for the Poisson equation with Dirichlet boundary conditions.
    """
    optimizer = optim.Adam(model.parameters(), lr=params["pinn_lr"])
    losses = []

    def loss_function(boundary_coefficient):
        # PDE residual at interior points
        res = poisson_residual(model, x_interior)
        loss_interior = torch.mean(res**2)

        # Boundary condition at boundary points
        u_pred_boundary = model(x_boundary)
        loss_boundary = torch.mean((u_pred_boundary - u_boundary) ** 2)

        # Total loss
        return loss_interior + boundary_coefficient * loss_boundary

    for epoch in range(params["pinn_epochs"]):
        optimizer.zero_grad()
        loss = loss_function(boundary_coefficient)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 500 == 0:
            print(f"Epoch {epoch} - Loss: {loss.item():.6f}")

    return losses


def train_supervised(model, X_train, y_train, X_val, y_val):
    """
    Train a supervised model using a reference solution.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params["sup_lr"])

    # Create DataLoader for batching
    dataset = torch.utils.data.TensorDataset(X_train.cpu(), y_train.cpu())
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=params["sup_batch_size"], shuffle=True
    )

    train_losses = []
    val_losses = []

    for epoch in range(params["sup_epochs"]):
        # Training
        model.train()
        epoch_train_loss = 0.0
        for batch_x, batch_y in dataloader:
            # Move batch to device
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item() * batch_x.size(0)

        epoch_train_loss /= len(dataset)
        train_losses.append(epoch_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            epoch_val_loss = val_loss.item()
        val_losses.append(epoch_val_loss)

        if epoch % 100 == 0:
            print(
                f"Epoch {epoch} - Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}"
            )

    return train_losses, val_losses
