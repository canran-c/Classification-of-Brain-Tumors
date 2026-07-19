from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm

from .config import ExperimentConfig, RuntimeConfig
from .data import create_dataloaders
from .models import build_model
from .utils import compute_multiclass_metrics


def train_one_experiment(runtime_cfg: RuntimeConfig, exp_cfg: ExperimentConfig) -> dict:
    train_loader, test_loader, classes = create_dataloaders(
        dataset_dir=runtime_cfg.dataset_dir,
        image_size=exp_cfg.image_size,
        use_preprocessing=exp_cfg.use_preprocessing,
        use_augmentation=exp_cfg.use_augmentation,
        batch_size=runtime_cfg.batch_size,
        num_workers=runtime_cfg.num_workers,
    )

    device = torch.device("cuda" if runtime_cfg.device == "auto" and torch.cuda.is_available() else runtime_cfg.device)
    if runtime_cfg.device == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")

    model = build_model(
        model_name=exp_cfg.model_name,
        num_classes=len(classes),
        use_pretrained=exp_cfg.use_pretrained,
    ).to(device)

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=runtime_cfg.learning_rate,
        weight_decay=runtime_cfg.weight_decay,
    )

    best_accuracy = -1.0
    best_state = None
    history = []

    for epoch in range(runtime_cfg.epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"{exp_cfg.experiment_id} | Epoch {epoch + 1}/{runtime_cfg.epochs}"):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        eval_metrics = evaluate(model, test_loader, device)
        eval_metrics["train_loss"] = float(train_loss)
        eval_metrics["epoch"] = epoch + 1
        history.append(eval_metrics)

        if eval_metrics["accuracy"] > best_accuracy:
            best_accuracy = eval_metrics["accuracy"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    final_metrics = evaluate(model, test_loader, device)
    final_metrics["classes"] = classes
    final_metrics["history"] = history

    checkpoint_dir = runtime_cfg.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{exp_cfg.experiment_id}.pt"
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    torch.save(state_dict, checkpoint_path)
    final_metrics["checkpoint"] = str(checkpoint_path)

    return final_metrics


@torch.no_grad()
def evaluate(model: nn.Module, data_loader, device: torch.device) -> dict:
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)
    return compute_multiclass_metrics(y_true=y_true, y_pred=y_pred, y_prob=y_prob)
