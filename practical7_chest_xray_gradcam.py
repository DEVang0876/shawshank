
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_score, recall_score, f1_score)

CLASSES = ("NORMAL", "PNEUMONIA")


def gradcam(model: nn.Module, target_layer: nn.Module, image: torch.Tensor, class_idx: int) -> np.ndarray:
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    forward_hook = target_layer.register_forward_hook(lambda _m, _i, output: activations.append(output))
    gradient_hook = target_layer.register_full_backward_hook(lambda _m, _gi, go: gradients.append(go[0]))
    try:
        model.zero_grad(set_to_none=True)
        score = model(image)[0, class_idx]
        score.backward()
        feature_map = activations[0][0]
        channel_weights = gradients[0][0].mean(dim=(1, 2))
        cam = F.relu((channel_weights[:, None, None] * feature_map).sum(dim=0))
        cam = F.interpolate(cam[None, None], size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy()
        span = cam.max() - cam.min()
        return (cam - cam.min()) / span if span > 1e-8 else np.zeros_like(cam)
    finally:
        forward_hook.remove(); gradient_hook.remove()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pretrained", action="store_true", help="Initialize with ImageNet weights (requires download)")
    parser.add_argument("--output-dir", default="./practical7_output")
    args = parser.parse_args()
    root = Path(args.data_dir)
    for split in ("train", "test"):
        for label in CLASSES:
            if not (root / split / label).is_dir():
                parser.error(f"Missing folder: {root / split / label}")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=3), transforms.Resize((224, 224)), transforms.ToTensor()])
    train_set = datasets.ImageFolder(root / "train", transform=transform)
    test_set = datasets.ImageFolder(root / "test", transform=transform)
    if train_set.classes != list(CLASSES) or test_set.classes != list(CLASSES):
        parser.error(f"Expected classes {CLASSES} in alphabetical order; found {train_set.classes}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False, num_workers=0)
    weights = models.ResNet18_Weights.DEFAULT if args.pretrained else None
    backbone = models.resnet18(weights=weights)
    backbone.fc = nn.Linear(backbone.fc.in_features, len(CLASSES))
    model = backbone.to(device)
    class_counts = np.bincount(train_set.targets, minlength=2)
    class_weights = torch.tensor(len(train_set) / (2.0 * np.maximum(class_counts, 1)), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    print(f"CHEST X-RAY CLASSIFIER | Student ID: 23AIML014 | Device: {device}")
    print(f"Train images: {len(train_set)} | Test images: {len(test_set)} | Class counts: {class_counts.tolist()}")
    started = time.time()
    for epoch in range(args.epochs):
        model.train(); total_loss = correct = seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images); loss = criterion(logits, labels)
            loss.backward(); optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item(); seen += len(labels)
        print(f"Epoch {epoch + 1}/{args.epochs}: loss={total_loss / seen:.4f}, accuracy={100 * correct / seen:.2f}%")
    model.eval(); truths: list[int] = []; predictions: list[int] = []; probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for images, labels in test_loader:
            probs = model(images.to(device)).softmax(dim=1).cpu().numpy()
            truths.extend(labels.numpy().tolist()); predictions.extend(probs.argmax(1).tolist()); probabilities.extend(probs)
    print(f"Test accuracy={accuracy_score(truths, predictions):.4f} | Precision={precision_score(truths, predictions, zero_division=0):.4f} | Recall={recall_score(truths, predictions, zero_division=0):.4f} | F1={f1_score(truths, predictions, zero_division=0):.4f}")
    print(classification_report(truths, predictions, labels=[0, 1], target_names=CLASSES, zero_division=0))
    cm = confusion_matrix(truths, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4)); im = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel="Predicted label", ylabel="True label", title="Chest X-ray confusion matrix")
    for i in range(2):
        for j in range(2): ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(out / "confusion_matrix.png", dpi=180); plt.close(fig)
    # Explain up to three samples from each available class.
    selected = []
    for label in (0, 1):
        selected.extend([i for i, (_, y) in enumerate(test_set.samples) if y == label][:3])
    target_layer = model.layer4[-1].conv2
    figure, axes = plt.subplots(max(len(selected), 1), 3, figsize=(10, 3.2 * max(len(selected), 1)), squeeze=False)
    for row_idx, sample_idx in enumerate(selected):
        image, true_label = test_set[sample_idx]
        batch = image.unsqueeze(0).to(device)
        with torch.no_grad():
            probs = model(batch).softmax(dim=1)[0].cpu().numpy()
        predicted = int(probs.argmax())
        heatmap = gradcam(model, target_layer, batch.clone().requires_grad_(True), predicted)
        gray = image[0].numpy()
        axes[row_idx, 0].imshow(gray, cmap="gray"); axes[row_idx, 0].set_title(f"Actual: {CLASSES[true_label]}")
        axes[row_idx, 1].imshow(heatmap, cmap="jet"); axes[row_idx, 1].set_title("Grad-CAM")
        axes[row_idx, 2].imshow(gray, cmap="gray"); axes[row_idx, 2].imshow(heatmap, cmap="jet", alpha=0.4)
        axes[row_idx, 2].set_title(f"Predicted: {CLASSES[predicted]} ({probs[predicted]:.1%})")
        for ax in axes[row_idx]: ax.axis("off")
    figure.tight_layout(); figure.savefig(out / "gradcam_examples.png", dpi=180); plt.close(figure)
    torch.save({"model_state": model.state_dict(), "classes": list(CLASSES)}, out / "resnet18_chest_xray.pt")
    print(f"Saved model, confusion matrix, and Grad-CAM examples to {out.resolve()} ({time.time() - started:.1f}s)")


if __name__ == "__main__":
    main()
