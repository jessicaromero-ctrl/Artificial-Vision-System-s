"""CNN con transfer learning (ResNet-18 preentrenada en ImageNet)."""

import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from metrics import evaluate_and_save

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DefectDataset(Dataset):
    def __init__(self, samples, class_to_idx, transform):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # Las imágenes NEU son en escala de grises; se replican a 3 canales
        # porque ResNet espera entrada RGB.
        img = Image.open(path).convert("RGB")
        return self.transform(img), self.class_to_idx[label]


def build_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(model, loader, criterion, optimizer, device):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def predict(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).argmax(1).cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(preds)


def run_cnn(
    train_samples,
    test_samples,
    classes,
    out_dir,
    epochs=15,
    img_size=224,
    batch_size=32,
    lr=3e-4,
    patience=4,
    val_size=0.1,
    seed=42,
    num_workers=0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_to_idx = {c: i for i, c in enumerate(classes)}
    train_tf, eval_tf = build_transforms(img_size)

    # Validación interna para early stopping; el conjunto de prueba no se toca aquí.
    tr_split, val_split = train_test_split(
        train_samples,
        test_size=val_size,
        stratify=[l for _, l in train_samples],
        random_state=seed,
    )
    train_loader = DataLoader(
        DefectDataset(tr_split, class_to_idx, train_tf),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    val_loader = DataLoader(
        DefectDataset(val_split, class_to_idx, eval_tf),
        batch_size=batch_size, num_workers=num_workers,
    )
    test_loader = DataLoader(
        DefectDataset(test_samples, class_to_idx, eval_tf),
        batch_size=batch_size, num_workers=num_workers,
    )

    model = build_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    best_val_loss, best_epoch, best_state, bad_epochs = float("inf"), 0, None, 0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step()
        history.append({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss, "val_acc": va_acc,
        })
        print(f"Época {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} "
              f"| val loss {va_loss:.4f} acc {va_acc:.3f}")

        if va_loss < best_val_loss:
            best_val_loss, best_epoch = va_loss, epoch
            best_state, bad_epochs = copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping en la época {epoch}.")
                break

    print(f"Mejor época según pérdida de validación: {best_epoch} "
          f"(val loss {best_val_loss:.4f})")
    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "best_resnet18.pt")
    (out_dir / "history_cnn.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    y_true, y_pred = predict(model, test_loader, device)
    metrics = evaluate_and_save(
        y_true, y_pred, classes, out_dir, name="cnn_resnet18",
        paths=[path for path, _ in test_samples],
    )
    metrics["best_val_loss"] = float(best_val_loss)
    metrics["best_epoch"] = int(best_epoch)
    metrics["early_stopping_monitor"] = "val_loss"
    (out_dir / "metrics_cnn_resnet18.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metrics["device"] = str(device)
    return metrics
