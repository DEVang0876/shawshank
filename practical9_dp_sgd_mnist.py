
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix


class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float, list[int], list[int]]:
    model.eval(); loss_sum = correct = total = 0; preds_all: list[int] = []; labels_all: list[int] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images); loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        preds = logits.argmax(1); correct += (preds == labels).sum().item(); total += len(labels)
        preds_all.extend(preds.cpu().tolist()); labels_all.extend(labels.cpu().tolist())
    model.train()
    return correct / total, loss_sum / total, preds_all, labels_all


def train_one(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, device: torch.device,
              epochs: int, name: str) -> list[dict]:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.08, momentum=0.9)
    history = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train(); loss_sum = seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images), labels)
            loss.backward(); optimizer.step()
            loss_sum += loss.item() * len(labels); seen += len(labels)
        accuracy, test_loss, _, _ = evaluate(model, test_loader, device)
        epsilon = 0.0
        history.append({"run": name, "epoch": epoch, "train_loss": loss_sum / max(seen, 1),
                        "test_accuracy": accuracy, "test_loss": test_loss, "epsilon": epsilon,
                        "delta": None,
                        "elapsed_seconds": time.time() - start})
        print(f"[{name}] epoch {epoch}/{epochs}: loss={loss_sum / max(seen, 1):.4f}, accuracy={accuracy:.2%} | no formal DP")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-dir", default="./practical9_output")
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    try:
        from opacus import PrivacyEngine
    except ImportError as exc:
        raise SystemExit("Opacus is required: pip install opacus") from exc
    random.seed(23); torch.manual_seed(23)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_data = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    baseline = DigitCNN().to(device)
    records.extend(train_one(baseline, train_loader, test_loader, device, args.epochs, "Baseline"))
    acc, _, preds, truths = evaluate(baseline, test_loader, device)
    print(f"\nBaseline final test accuracy: {acc:.2%} (no formal privacy guarantee)")
    print(classification_report(truths, preds, digits=4, zero_division=0))
    cm = confusion_matrix(truths, preds, labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(8, 7)); image = ax.imshow(cm, cmap="Blues")
    ax.set(xlabel="Predicted digit", ylabel="Actual digit", title="Baseline MNIST confusion matrix")
    fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(output / "baseline_confusion_matrix.png", dpi=180); plt.close(fig)

    delta = 1e-5
    for target_epsilon in (1.0, 10.0):
        model = DigitCNN().to(device)
        engine = PrivacyEngine()
        # Calibrate noise for the requested total epsilon over this exact run.
        model, optimizer, private_loader = engine.make_private_with_epsilon(
            module=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.08, momentum=0.9),
            data_loader=train_loader, epochs=args.epochs, target_epsilon=target_epsilon,
            target_delta=delta, max_grad_norm=1.0)
        # train_one can optionally privatize itself; this model/optimizer/loader are already wrapped.
        name = f"DP-SGD target epsilon {target_epsilon:g}"
        history = []
        start = time.time()
        for epoch in range(1, args.epochs + 1):
            model.train(); loss_sum = seen = 0
            for images, labels in private_loader:
                if len(images) == 0:
                    continue
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True); loss = F.cross_entropy(model(images), labels)
                loss.backward(); optimizer.step(); loss_sum += loss.item() * len(labels); seen += len(labels)
            test_acc, test_loss, _, _ = evaluate(model, test_loader, device)
            eps = engine.get_epsilon(delta)
            history.append({"run": name, "epoch": epoch, "train_loss": loss_sum / max(seen, 1),
                            "test_accuracy": test_acc, "test_loss": test_loss, "epsilon": eps,
                            "delta": delta, "elapsed_seconds": time.time() - start})
            print(f"[{name}] epoch {epoch}/{args.epochs}: loss={loss_sum / max(seen, 1):.4f}, accuracy={test_acc:.2%}, epsilon={eps:.3f}")
        records.extend(history)
    report = pd.DataFrame(records)
    report.to_csv(output / "privacy_training_summary.csv", index=False)
    print("\nFinal privacy and utility comparison:")
    print(report.groupby("run", sort=False).tail(1)[["run", "test_accuracy", "epsilon", "delta"]].to_string(index=False))
    print(f"Saved results to {output.resolve()}")


if __name__ == "__main__":
    main()
