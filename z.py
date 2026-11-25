import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

STUDENT = "Devang | 23AIML014"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Step 1: Data (MNIST)
tfm = transforms.ToTensor()
train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def train(model, loader, epochs=1, adv_train=False, eps=0.2):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if adv_train:
                x = fgsm_attack(model, x, y, eps)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()


def accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100 * correct / total


def fgsm_attack(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    grad = torch.autograd.grad(loss, x)[0]
    x_adv = (x + eps * grad.sign()).clamp(0, 1)
    return x_adv.detach()


def pgd_attack(model, x, y, eps=0.2, alpha=0.02, steps=10):
    x_orig = x.clone().detach()
    x_adv = x.clone().detach()
    for _ in range(steps):
        x_adv.requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x_orig - eps), x_orig + eps).clamp(0, 1)
    return x_adv.detach()


def adv_accuracy(model, loader, attack_fn, **kwargs):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = attack_fn(model, x, y, **kwargs)
        pred = model(x_adv).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100 * correct / total


def main():
    print(f"[{STUDENT}]")

    baseline = SmallCNN().to(device)
    train(baseline, train_loader, epochs=1)

    clean_acc = accuracy(baseline, test_loader)
    fgsm_acc = adv_accuracy(baseline, test_loader, fgsm_attack, eps=0.2)
    pgd_acc = adv_accuracy(baseline, test_loader, pgd_attack, eps=0.2, alpha=0.02, steps=10)

    print(f"Baseline clean accuracy: {clean_acc:.2f}%")
    print(f"Baseline FGSM accuracy: {fgsm_acc:.2f}%")
    print(f"Baseline PGD accuracy: {pgd_acc:.2f}%")

    adv_model = SmallCNN().to(device)
    train(adv_model, train_loader, epochs=1, adv_train=True, eps=0.2)

    adv_clean = accuracy(adv_model, test_loader)
    adv_fgsm = adv_accuracy(adv_model, test_loader, fgsm_attack, eps=0.2)
    adv_pgd = adv_accuracy(adv_model, test_loader, pgd_attack, eps=0.2, alpha=0.02, steps=10)

    print(f"Adversarially trained clean accuracy: {adv_clean:.2f}%")
    print(f"Adversarially trained FGSM accuracy: {adv_fgsm:.2f}%")
    print(f"Adversarially trained PGD accuracy: {adv_pgd:.2f}%")


if __name__ == "__main__":
    main()