"""Waterbirds: Dense ResNet18 vs. RigL-Sparse ResNet18 (official sparse_learning library).

Uses TimDettmers/sparse_learning (cloned into ./sparse_learning), the PyTorch
library behind "Sparse Networks from Scratch" that RigL/SET/ITOP-style follow-up
papers build on. We drive it with:
  --growth gradient --prune magnitude --redistribution none
which reproduces RigL's actual criterion (prune smallest-magnitude weights,
grow where the instantaneous gradient magnitude is largest), as opposed to the
library's own default "momentum growth" (Sparse Networks From Scratch) or
"random growth" (SET).

Usage:
    pip install -r requirements.txt
    python train_waterbirds.py --epochs 5 --density 0.2 --method gradient
    python train_waterbirds.py --epochs 5 --density 0.2 --method random   # SET
    python train_waterbirds.py --epochs 5 --dense                        # baseline
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sparse_learning"))

import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as transforms
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

from sparselearning.core import Masking, CosineDecay


def build_model():
    return tvm.resnet18(weights=None, num_classes=2)


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    group_correct, group_total = {}, {}
    with torch.no_grad():
        for x, y, metadata in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            place = metadata[:, 0]
            for yi, pi, ci in zip(y.tolist(), place.tolist(), (pred == y).tolist()):
                key = (yi, pi)
                group_total[key] = group_total.get(key, 0) + 1
                group_correct[key] = group_correct.get(key, 0) + ci
    acc = correct / total
    worst = min(group_correct[k] / group_total[k] for k in group_total)
    return acc, worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--density", type=float, default=0.2, help="1 - sparsity")
    parser.add_argument("--prune_rate", type=float, default=0.3)
    parser.add_argument("--method", type=str, default="gradient",
                         choices=["gradient", "random", "momentum"],
                         help="gradient=RigL, random=SET, momentum=Sparse Networks From Scratch")
    parser.add_argument("--dense", action="store_true", help="skip sparsity, train dense baseline")
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = get_dataset(dataset="waterbirds", download=True, root_dir=args.data_dir)
    train_loader = get_train_loader(
        "standard", dataset.get_subset("train", transform=TRANSFORM), batch_size=args.batch_size)
    eval_loader = get_eval_loader(
        "standard", dataset.get_subset("test", transform=TRANSFORM), batch_size=args.batch_size)

    model = build_model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    mask = None
    if not args.dense:
        decay = CosineDecay(args.prune_rate, len(train_loader) * args.epochs)
        mask = Masking(
            optimizer, decay,
            prune_rate=args.prune_rate,
            prune_mode="magnitude",
            growth_mode=args.method,
            redistribution_mode="none",
            prune_every_k_steps=len(train_loader),  # regrow once per epoch
        )
        mask.add_module(model, density=args.density)

    tag = "DENSE" if args.dense else f"SPARSE(density={args.density}, growth={args.method})"
    print(f"=== {tag} ===")

    for epoch in range(args.epochs):
        model.train()
        for x, y, metadata in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            if mask is not None:
                mask.step()  # optimizer.step() + mask apply + periodic prune/grow
            else:
                optimizer.step()

        acc, worst = evaluate(model, eval_loader, device)
        print(f"[epoch {epoch+1}/{args.epochs}] test_acc={acc:.3f} worst_group_acc={worst:.3f}")


if __name__ == "__main__":
    main()
