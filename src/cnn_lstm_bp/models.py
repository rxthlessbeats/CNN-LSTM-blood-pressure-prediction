import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


def load_checkpoint(path, device="cpu"):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


class BP_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=2, out_channels=128, kernel_size=3, padding=1, bias=True
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=10, stride=2)

        self.conv2 = nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=True
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=10, stride=2)

        self.conv3 = nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=True
        )
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=2)

        self.lstm1 = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.fc = nn.Linear(256, 1)

        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = x.permute(0, 2, 1)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)

        x = x[:, -1, :]

        x = self.fc(x)

        return x


class BP_Model_DO(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=2, out_channels=128, kernel_size=3, padding=1, bias=True
        )
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=10, stride=2)

        self.conv2 = nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=3, padding=1, bias=True
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=10, stride=2)

        self.lstm1 = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)

        x, _ = self.lstm1(x)
        x = self.dropout(x)
        x, _ = self.lstm2(x)
        x = self.dropout(x)

        x = x[:, -1, :]

        x = self.fc1(x)
        x = self.fc2(x)
        return x


def train_torch_model(
    model,
    train_loader,
    val_loader,
    num_epochs,
    criterion,
    optimizer,
    device,
    bp_type,
    scheduler=None,
    early_stopper=None,
    ckpt_path=None,
):
    model.to(device)
    path = ckpt_path or f"model/torch_cnn_lstm_{bp_type}.pth"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    train_losses = []
    val_losses = []
    train_mae_losses = []
    val_mae_losses = []
    best_val_loss = float("inf")
    epoch = -1

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_mae = 0.0

        with tqdm(
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
            unit="batch",
        ) as pbar:
            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                mae_loss = F.l1_loss(outputs, targets)

                loss.backward()
                optimizer.step()

                batch_size = inputs.size(0)
                epoch_train_loss += loss.item() * batch_size
                epoch_train_mae += mae_loss.item() * batch_size

                pbar.set_postfix({"loss": loss.item(), "mae": mae_loss.item()})
                pbar.update(1)

        epoch_train_loss /= len(train_loader.dataset)
        epoch_train_mae /= len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        train_mae_losses.append(epoch_train_mae)

        model.eval()
        epoch_val_loss = 0.0
        epoch_val_mae = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                mae_loss = F.l1_loss(outputs, targets)

                batch_size = inputs.size(0)
                epoch_val_loss += loss.item() * batch_size
                epoch_val_mae += mae_loss.item() * batch_size

        epoch_val_loss /= len(val_loader.dataset)
        epoch_val_mae /= len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        val_mae_losses.append(epoch_val_mae)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), path)

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{num_epochs} - "
            f"Training Loss: {epoch_train_loss:.4f}, MAE: {epoch_train_mae:.4f}, "
            f"Validation Loss: {epoch_val_loss:.4f}, MAE: {epoch_val_mae:.4f}, "
            f"Learning Rate: {current_lr:.6f}"
        )

        if scheduler:
            scheduler.step(epoch_val_loss)

        if early_stopper and early_stopper.step(epoch_val_loss):
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    if os.path.exists(path):
        model.load_state_dict(load_checkpoint(path, device))

    return train_losses, val_losses, train_mae_losses, val_mae_losses, epoch + 1


def evaluate_torch_model(model, test_loader, device):
    model.eval()
    model.to(device)

    predictions = []
    targets_list = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            predictions.extend(outputs.cpu().numpy())
            targets_list.extend(targets.cpu().numpy())

    return np.array(predictions), np.array(targets_list)


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0

    def step(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True

        return False
