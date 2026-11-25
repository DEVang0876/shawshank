
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

SEED = 23
TARGET_LABEL = 0


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def add_trigger(image: torch.Tensor) -> torch.Tensor:
    """Add a small, fixed white square in the lower-right corner."""
    out = image.clone()
    out[..., -4:-1, -4:-1] = 1.0
    return out


class PoisonedMNIST(Dataset):
    def __init__(self, source: Dataset, poison_rate: float, target: int = TARGET_LABEL):
        if not 0.0 <= poison_rate <= 1.0:
            raise ValueError("poison_rate must be between 0 and 1")
        self.source, self.target = source, target
        count = int(len(source) * poison_rate)
        rng = np.random.default_rng(SEED)
        self.poison_indices = set(rng.choice(len(source), count, replace=False).tolist())

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, index: int):
        image, label = self.source[index]
        if index in self.poison_indices:
            return add_trigger(image), self.target
        return image, label


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.fc1(x.flatten(1)))
        return self.fc2(self.dropout(x))


def train(model: nn.Module, loader: DataLoader, device: torch.device, epochs: int) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        model.train()
        total_loss, correct, seen = 0.0, 0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            seen += len(labels)
        print(f"Epoch {epoch + 1}/{epochs}: loss={total_loss / seen:.4f}, accuracy={100 * correct / seen:.2f}%")


@torch.inference_mode()
def accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
        total += len(labels)
    return 100.0 * correct / total


@torch.inference_mode()
def attack_success_rate(model: nn.Module, dataset: Dataset, device: torch.device,
                        target: int = TARGET_LABEL, batch_size: int = 256) -> float:
    model.eval()
    triggered = [(add_trigger(x), y) for x, y in dataset if y != target]
    loader = DataLoader(triggered, batch_size=batch_size)
    hits = total = 0
    for images, _ in loader:
        predictions = model(images.to(device)).argmax(1)
        hits += (predictions == target).sum().item()
        total += len(images)
    return 100.0 * hits / total if total else 0.0


def channel_trigger_scores(model: SmallCNN, loader: DataLoader, device: torch.device) -> torch.Tensor:
    """Rank conv2 channels by their mean activation increase on triggered inputs."""
    model.eval()
    sums = torch.zeros(model.conv2.out_channels, device=device)
    batches = 0
    for images, _ in loader:
        images = images.to(device)
        triggered = images.clone()
        triggered[..., -4:-1, -4:-1] = 1.0
        with torch.no_grad():
            clean_map = F.relu(model.conv2(F.max_pool2d(F.relu(model.conv1(images)), 2)))
            trigger_map = F.relu(model.conv2(F.max_pool2d(F.relu(model.conv1(triggered)), 2)))
        sums += (trigger_map - clean_map).mean(dim=(0, 2, 3))
        batches += 1
    return sums / max(batches, 1)


def prune_channels(model: SmallCNN, channel_scores: torch.Tensor, fraction: float) -> None:
    count = min(int(model.conv2.out_channels * fraction), model.conv2.out_channels)
    if count <= 0:
        return
    indices = torch.argsort(channel_scores, descending=True)[:count]
    with torch.no_grad():
        model.conv2.weight[indices] = 0
        if model.conv2.bias is not None:
            model.conv2.bias[indices] = 0
        model.fc1.weight.view(128, 64, 7, 7)[:, indices] = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--poison-rate", type=float, default=0.10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./practical4_output")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    poisoned = PoisonedMNIST(train_data, args.poison_rate)
    train_loader = DataLoader(poisoned, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)
    model = SmallCNN().to(device)
    print(f"Student ID: 23AIML014 | Device: {device} | Poisoned training examples: {len(poisoned.poison_indices)}")
    started = time.time()
    train(model, train_loader, device, args.epochs)
    clean_acc = accuracy(model, test_loader, device)
    asr = attack_success_rate(model, test_data, device)
    scores = channel_trigger_scores(model, test_loader, device)
    print(f"Before pruning: clean accuracy={clean_acc:.2f}% | backdoor success={asr:.2f}%")
    fractions = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    rows = []
    for fraction in fractions:
        candidate = SmallCNN().to(device)
        candidate.load_state_dict(model.state_dict())
        prune_channels(candidate, scores, fraction)
        rows.append((fraction * 100, accuracy(candidate, test_loader, device),
                     attack_success_rate(candidate, test_data, device)))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "pruning_results.csv", np.asarray(rows), delimiter=",",
               header="channels_pruned_percent,clean_accuracy_percent,attack_success_percent", comments="")
    percentages, clean, attack = zip(*rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(percentages, clean, marker="o", label="Clean accuracy")
    ax.plot(percentages, attack, marker="s", label="Backdoor success")
    ax.set(xlabel="Conv2 channels pruned (%)", ylabel="Rate (%)", ylim=(0, 100),
           title="Channel pruning: utility and backdoor behavior")
    ax.grid(alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(out / "pruning_tradeoff.png", dpi=180); plt.close(fig)
    torch.save(model.state_dict(), out / "backdoored_mnist_cnn.pt")
    print("Pruning results (% pruned | clean accuracy | attack success):")
    for row in rows:
        print(f"{row[0]:5.0f}% | {row[1]:6.2f}% | {row[2]:6.2f}%")
    print(f"Saved artifacts to {out.resolve()} (elapsed {time.time() - started:.1f}s)")


if __name__ == "__main__":
    main()
