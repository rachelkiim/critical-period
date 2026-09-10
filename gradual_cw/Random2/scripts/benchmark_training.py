#!/usr/bin/env python3
"""Benchmark training: CNN with and without random noise warmup on CIFAR-10."""

import sys
import os
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np

import torchvision

from src.data.dataloader import get_cifar10_transform
from src.training.training import train, validation
from src.training.random_training import random_train, random_validation
from src.models import get_model, MODEL_REGISTRY
from src.logger import NumpyEncoder


def parse_args():
    models = list(MODEL_REGISTRY.keys())
    parser = argparse.ArgumentParser(description="Benchmark CNN training")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=models,
                        help=f"Model architecture (default: resnet18)")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50,
                        help="Downstream training epochs")
    parser.add_argument("--epochs_noise", type=int, default=5,
                        help="Random noise warmup epochs")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_noise", type=int, default=50000,
                        help="Number of random noise samples per warmup epoch")
    parser.add_argument("--data_path", type=str, default="./data")
    parser.add_argument("--output_path", type=str, default="./exp")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Direct output directory (skip auto exp_id generation)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Determine output directory
    if args.output_dir:
        exp_path = args.output_dir
        exp_id = os.path.basename(os.path.normpath(exp_path))
    else:
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        exp_id = f"{timestamp}_benchmark_{args.model}_ep{args.epochs}_wn{args.epochs_noise}"
        exp_path = os.path.join(args.output_path, exp_id)

    checkpoint_path = os.path.join(exp_path, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)

    # Save config
    config = vars(args)
    config["exp_id"] = exp_id
    config["exp_path"] = exp_path
    with open(os.path.join(exp_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Load CIFAR-10
    train_transform = get_cifar10_transform(augmentation=True)
    test_transform = get_cifar10_transform(augmentation=False)

    cifar10_train = torchvision.datasets.CIFAR10(
        root=args.data_path, train=True, download=True, transform=train_transform)
    cifar10_test = torchvision.datasets.CIFAR10(
        root=args.data_path, train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        cifar10_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        cifar10_test, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Create models
    model_w = get_model(args.model, num_classes=args.num_classes).to(device)
    model_wo = get_model(args.model, num_classes=args.num_classes).to(device)

    loss_fn = nn.CrossEntropyLoss()
    input_shape = (3, 32, 32)

    info_w = {"warmup_loss": [], "warmup_acc": [],
              "train_loss": [], "train_acc": [],
              "test_loss": [], "test_acc": []}
    info_wo = {"train_loss": [], "train_acc": [],
               "test_loss": [], "test_acc": []}

    # Random noise warmup for model_w
    optimizer_warmup = torch.optim.SGD(
        model_w.parameters(), lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay)

    print("Starting Random Noise Warmup...")
    for epoch in range(args.epochs_noise + 1):
        if epoch == 0:
            loss, acc = random_validation(
                model_w, loss_fn, input_shape, args.num_noise,
                args.batch_size, args.num_classes, mean=0, std=1, device=device)
        else:
            loss, acc = random_train(
                model_w, optimizer_warmup, loss_fn, input_shape,
                args.num_noise, args.batch_size, args.num_classes,
                mean=0, std=1, device=device)
        info_w["warmup_loss"].append(loss)
        info_w["warmup_acc"].append(acc)
        print(f"  Warmup Epoch {epoch}/{args.epochs_noise}: "
              f"Loss={loss:.4f}, Acc={acc:.4f}")

    # Downstream training
    optimizer_w = torch.optim.SGD(
        model_w.parameters(), lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_wo = torch.optim.SGD(
        model_wo.parameters(), lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler_w = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_w, T_max=args.epochs)
    scheduler_wo = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_wo, T_max=args.epochs)

    # Train model_wo (without warmup)
    print("\nTraining model without warmup...")
    for epoch in range(args.epochs + 1):
        if epoch == 0:
            train_loss, train_acc = validation(
                model_wo, train_loader, loss_fn, device=device)
        else:
            train_loss, train_acc = train(
                model_wo, train_loader, optimizer_wo, loss_fn,
                scheduler_wo, device=device)
        test_loss, test_acc = validation(model_wo, test_loader, loss_fn, device=device)
        info_wo["train_loss"].append(train_loss)
        info_wo["train_acc"].append(train_acc)
        info_wo["test_loss"].append(test_loss)
        info_wo["test_acc"].append(test_acc)
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch}/{args.epochs}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

    # Train model_w (with warmup)
    print("\nTraining model with warmup...")
    for epoch in range(args.epochs + 1):
        if epoch == 0:
            train_loss, train_acc = validation(
                model_w, train_loader, loss_fn, device=device)
        else:
            train_loss, train_acc = train(
                model_w, train_loader, optimizer_w, loss_fn,
                scheduler_w, device=device)
        test_loss, test_acc = validation(model_w, test_loader, loss_fn, device=device)
        info_w["train_loss"].append(train_loss)
        info_w["train_acc"].append(train_acc)
        info_w["test_loss"].append(test_loss)
        info_w["test_acc"].append(test_acc)
        if epoch % 5 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch}/{args.epochs}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

    # Save checkpoints
    torch.save(model_wo.state_dict(), os.path.join(checkpoint_path, "model_wo.pth"))
    torch.save(model_w.state_dict(), os.path.join(checkpoint_path, "model_w.pth"))

    # Save training results
    results = {"with_warmup": info_w, "without_warmup": info_wo}
    with open(os.path.join(exp_path, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"\nExperiment complete: {exp_id}")
    print(f"Results saved to: {exp_path}")
    print(f"EXP_ID={exp_id}")
    print(f"EXP_PATH={exp_path}")


if __name__ == "__main__":
    main()
