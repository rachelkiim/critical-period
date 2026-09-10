#!/usr/bin/env python3
"""ResNet18 experiment with selectable init mode and random-noise warmup on CIFAR-10."""

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
from torchvision.models import resnet18, ResNet18_Weights

from src.training.training import train, validation
from src.training.random_training import random_train, random_validation
from src.evaluation.inference import inference, get_confidence_only
from src.evaluation.calibration import reliability_diagram, ece
from src.evaluation.ood_detection import roc_curve, auc
from src.logger import NumpyEncoder

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser(description="ResNet18 experiment on CIFAR-10")
    parser.add_argument("--init_mode", type=str, default="pretrained",
                        choices=["pretrained", "untrained"],
                        help="Model initialization mode (default: pretrained)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20,
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


def create_pretrained_resnet18(num_classes):
    """Load pretrained ResNet18 and replace FC layer for num_classes classification.

    Applies He (Kaiming) initialization to the new FC layer.
    """
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    nn.init.kaiming_normal_(model.fc.weight, mode='fan_out', nonlinearity='relu')
    nn.init.zeros_(model.fc.bias)
    return model


def create_untrained_resnet18(num_classes):
    """Create ResNet18 with random initialization and replace FC layer."""
    model = resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return model


def freeze_backbone(model):
    """Freeze all parameters except the FC layer."""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_all(model):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True


def get_cifar10_imagenet_transform(augmentation=False):
    """Get CIFAR-10 transform pipeline resized to ImageNet input size (224x224).

    Uses ImageNet normalization to match pretrained ResNet18 expectations.
    """
    transforms = [torchvision.transforms.Resize(224)]
    if augmentation:
        transforms.append(torchvision.transforms.RandomCrop(224, padding=28))
        transforms.append(torchvision.transforms.RandomHorizontalFlip())
    transforms.append(torchvision.transforms.ToTensor())
    transforms.append(torchvision.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return torchvision.transforms.Compose(transforms)


def get_svhn_imagenet_transform():
    """Get SVHN transform pipeline resized to ImageNet input size (224x224)."""
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Init mode: {args.init_mode}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Determine output directory
    if args.output_dir:
        exp_path = args.output_dir
        exp_id = os.path.basename(os.path.normpath(exp_path))
    else:
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        exp_id = f"{timestamp}_resnet_{args.init_mode}_ep{args.epochs}_wn{args.epochs_noise}"
        exp_path = os.path.join(args.output_path, exp_id)
    checkpoint_path = os.path.join(exp_path, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)

    # Save config
    config = vars(args)
    config["exp_id"] = exp_id
    config["model"] = f"resnet18_{args.init_mode}"
    with open(os.path.join(exp_path, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Load CIFAR-10 resized to 224x224 with ImageNet normalization
    train_transform = get_cifar10_imagenet_transform(augmentation=True)
    test_transform = get_cifar10_imagenet_transform(augmentation=False)

    cifar10_train = torchvision.datasets.CIFAR10(
        root=args.data_path, train=True, download=True, transform=train_transform)
    cifar10_test = torchvision.datasets.CIFAR10(
        root=args.data_path, train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        cifar10_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(
        cifar10_test, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Load SVHN (OOD) resized to 224x224 with ImageNet normalization
    svhn_transform = get_svhn_imagenet_transform()
    svhn_test = torchvision.datasets.SVHN(
        root=args.data_path, split='test', download=True, transform=svhn_transform)
    ood_loader = torch.utils.data.DataLoader(
        svhn_test, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Create models based on initialization mode
    if args.init_mode == "pretrained":
        model_w = create_pretrained_resnet18(args.num_classes).to(device)
        model_wo = create_pretrained_resnet18(args.num_classes).to(device)
    else:
        model_w = create_untrained_resnet18(args.num_classes).to(device)
        model_wo = create_untrained_resnet18(args.num_classes).to(device)

    loss_fn = nn.CrossEntropyLoss()
    input_shape = (3, 224, 224)  # ImageNet input size

    info_w = {"warmup_loss": [], "warmup_acc": [],
              "train_loss": [], "train_acc": [],
              "test_loss": [], "test_acc": []}
    info_wo = {"train_loss": [], "train_acc": [],
               "test_loss": [], "test_acc": []}

    # =========================================================================
    # Random noise warmup for model_w
    # =========================================================================
    if args.init_mode == "pretrained":
        freeze_backbone(model_w)
        warmup_params = model_w.fc.parameters()
        warmup_desc = "FC layer only"
    else:
        warmup_params = model_w.parameters()
        warmup_desc = "all parameters"

    optimizer_warmup = torch.optim.SGD(
        warmup_params, lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay)

    print(f"Starting Random Noise Warmup ({warmup_desc})...")
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

    # Unfreeze all parameters for downstream training
    if args.init_mode == "pretrained":
        unfreeze_all(model_w)

    # =========================================================================
    # Downstream training
    # =========================================================================
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

    # =========================================================================
    # Calibration evaluation (basic ECE)
    # =========================================================================
    print("\n=== Calibration Evaluation ===")

    pred_wo, conf_wo = inference(model_wo, test_loader, device=device)
    pred_w, conf_w = inference(model_w, test_loader, device=device)

    acc_bins_wo, conf_bins_wo, ns_wo = reliability_diagram(pred_wo, conf_wo)
    acc_bins_w, conf_bins_w, ns_w = reliability_diagram(pred_w, conf_w)

    ece_wo = ece(acc_bins_wo, conf_bins_wo, ns_wo)
    ece_w = ece(acc_bins_w, conf_bins_w, ns_w)

    print(f"ECE (w/o warmup): {ece_wo:.4f}")
    print(f"ECE (w/ warmup):  {ece_w:.4f}")

    # =========================================================================
    # OOD detection evaluation (basic AUROC)
    # =========================================================================
    print("\n=== OOD Detection Evaluation ===")

    _, conf_id_wo = inference(model_wo, test_loader, device=device)
    conf_ood_wo = get_confidence_only(model_wo, ood_loader, device=device)
    fp_wo, tp_wo = roc_curve(conf_id_wo, conf_ood_wo)
    auroc_wo = auc(fp_wo, tp_wo)

    _, conf_id_w = inference(model_w, test_loader, device=device)
    conf_ood_w = get_confidence_only(model_w, ood_loader, device=device)
    fp_w, tp_w = roc_curve(conf_id_w, conf_ood_w)
    auroc_w = auc(fp_w, tp_w)

    print(f"AUROC (w/o warmup): {auroc_wo:.4f}")
    print(f"AUROC (w/ warmup):  {auroc_w:.4f}")

    # =========================================================================
    # Save results
    # =========================================================================
    results = {
        "with_warmup": {
            **info_w,
            "ece": float(ece_w),
            "auroc": float(auroc_w),
            "reliability_diagram": {
                "acc": acc_bins_w.tolist() if hasattr(acc_bins_w, 'tolist') else acc_bins_w,
                "conf_bin": conf_bins_w.tolist() if hasattr(conf_bins_w, 'tolist') else conf_bins_w,
                "num_sample": ns_w.tolist() if hasattr(ns_w, 'tolist') else ns_w,
            },
            "roc": {
                "fp": fp_w.tolist() if hasattr(fp_w, 'tolist') else fp_w,
                "tp": tp_w.tolist() if hasattr(tp_w, 'tolist') else tp_w,
            },
        },
        "without_warmup": {
            **info_wo,
            "ece": float(ece_wo),
            "auroc": float(auroc_wo),
            "reliability_diagram": {
                "acc": acc_bins_wo.tolist() if hasattr(acc_bins_wo, 'tolist') else acc_bins_wo,
                "conf_bin": conf_bins_wo.tolist() if hasattr(conf_bins_wo, 'tolist') else conf_bins_wo,
                "num_sample": ns_wo.tolist() if hasattr(ns_wo, 'tolist') else ns_wo,
            },
            "roc": {
                "fp": fp_wo.tolist() if hasattr(fp_wo, 'tolist') else fp_wo,
                "tp": tp_wo.tolist() if hasattr(tp_wo, 'tolist') else tp_wo,
            },
        },
    }

    with open(os.path.join(exp_path, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"\n=== Final Summary ===")
    print(f"Test Acc  (w/o): {info_wo['test_acc'][-1]:.4f}  |  (w/): {info_w['test_acc'][-1]:.4f}")
    print(f"ECE       (w/o): {ece_wo:.4f}  |  (w/): {ece_w:.4f}")
    print(f"AUROC     (w/o): {auroc_wo:.4f}  |  (w/): {auroc_w:.4f}")
    print(f"\nExperiment complete: {exp_id}")
    print(f"Results saved to: {exp_path}")


if __name__ == "__main__":
    main()
