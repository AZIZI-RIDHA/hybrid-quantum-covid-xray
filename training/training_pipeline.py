import torch
import torch.optim as optim

def train_model(model, train_loader, val_loader, config):
    """
    Training loop matching the paper methodology (Section IV-C)
    """
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = torch.nn.BCELoss()

    best_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(config["epochs"]):
        # -------- Training --------
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x).squeeze()
            loss = criterion(outputs, y.float())
            loss.backward()
            optimizer.step()

        # -------- Validation --------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                outputs = model(x).squeeze()
                val_loss += criterion(outputs, y.float()).item()

        val_loss /= len(val_loader)

        # -------- Early Stopping --------
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1

        if patience_counter >= config["early_stopping"]:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model