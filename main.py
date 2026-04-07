#!/usr/bin/env python3
import argparse
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


class LeNet5(nn.Module):
    def __init__(self, c1_channels: int = 12) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, c1_channels, 5)
        self.conv2 = nn.Conv2d(c1_channels, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.bn1 = nn.BatchNorm2d(c1_channels)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm1d(120)
        self.bn5 = nn.BatchNorm1d(84)

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.relu(self.bn4(self.fc1(x)))
        x = self.relu(self.bn5(self.fc2(x)))
        return self.fc3(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast MNIST benchmark trainer.")
    parser.add_argument("--data-dir", type=str, default="~/.pytorch/MNIST_data/")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.022)
    parser.add_argument("--weight-decay", type=float, default=2e-5)
    parser.add_argument("--pct-start", type=float, default=0.24)
    parser.add_argument("--div-factor", type=float, default=18.0)
    parser.add_argument("--final-div-factor", type=float, default=1e5)
    parser.add_argument("--c1-channels", type=int, default=16)
    parser.add_argument("--label-smoothing", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-time", type=float, default=10.4)
    parser.add_argument("--target-acc", type=float, default=99.3)
    parser.add_argument("--tta-shifts", type=int, choices=[0, 5, 9], default=0)
    parser.add_argument(
        "--in-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep MNIST tensors on the training device and iterate by index.",
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def maybe_sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def evaluate(
    model: nn.Module,
    testloader: torch.utils.data.DataLoader,
    device: torch.device,
    non_blocking: bool,
    tta_shifts: int = 0,
) -> float:
    if tta_shifts == 0:
        shifts = [(0, 0)]
    elif tta_shifts == 5:
        shifts = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    else:
        shifts = [
            (0, 0),
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]

    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            if len(shifts) == 1:
                logits = model(images)
            else:
                logits = 0.0
                for dy, dx in shifts:
                    logits = logits + model(
                        torch.roll(images, shifts=(dy, dx), dims=(2, 3))
                    )
                logits = logits / len(shifts)
            pred = logits.argmax(1)
            correct += (pred == labels).sum().item()
    return 100.0 * correct / len(testloader.dataset)


def mnist_to_tensors(dataset: datasets.MNIST) -> tuple[torch.Tensor, torch.Tensor]:
    images = dataset.data.float().div_(255.0)
    images = images.sub_(0.1307).div_(0.3081).unsqueeze(1)
    labels = dataset.targets.long()
    return images, labels


def evaluate_tensors(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    eval_batch_size: int,
    tta_shifts: int = 0,
) -> float:
    if tta_shifts == 0:
        shifts = [(0, 0)]
    elif tta_shifts == 5:
        shifts = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    else:
        shifts = [
            (0, 0),
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]

    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(images), eval_batch_size):
            x = images[start : start + eval_batch_size]
            y = labels[start : start + eval_batch_size]
            if len(shifts) == 1:
                logits = model(x)
            else:
                logits = 0.0
                for dy, dx in shifts:
                    logits = logits + model(torch.roll(x, shifts=(dy, dx), dims=(2, 3)))
                logits = logits / len(shifts)
            correct += (logits.argmax(1) == y).sum().item()
    return 100.0 * correct / len(labels)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    device = get_device()
    non_blocking = device.type in {"cuda", "mps"}
    pin_memory = device.type == "cuda"
    data_dir = os.path.expanduser(args.data_dir)

    if args.in_memory:
        trainset = datasets.MNIST(data_dir, download=True, train=True)
        testset = datasets.MNIST(data_dir, download=True, train=False)
        train_images, train_labels = mnist_to_tensors(trainset)
        test_images, test_labels = mnist_to_tensors(testset)
        train_images = train_images.to(device, non_blocking=non_blocking)
        train_labels = train_labels.to(device, non_blocking=non_blocking)
        test_images = test_images.to(device, non_blocking=non_blocking)
        test_labels = test_labels.to(device, non_blocking=non_blocking)
        steps_per_epoch = len(train_images) // args.batch_size
    else:
        train_tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        test_tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        trainset = datasets.MNIST(data_dir, download=True, train=True, transform=train_tf)
        testset = datasets.MNIST(data_dir, download=True, train=False, transform=test_tf)

        train_loader_kwargs: dict[str, object] = {
            "batch_size": args.batch_size,
            "shuffle": True,
            "num_workers": args.num_workers,
            "pin_memory": pin_memory,
        }
        test_loader_kwargs: dict[str, object] = {
            "batch_size": args.batch_size * 2,
            "shuffle": False,
            "num_workers": args.num_workers,
            "pin_memory": pin_memory,
        }

        if args.num_workers > 0:
            train_loader_kwargs["persistent_workers"] = True
            train_loader_kwargs["prefetch_factor"] = 4
            test_loader_kwargs["persistent_workers"] = True
            test_loader_kwargs["prefetch_factor"] = 2
            if sys.platform == "darwin":
                train_loader_kwargs["multiprocessing_context"] = "fork"
                test_loader_kwargs["multiprocessing_context"] = "fork"

        trainloader = torch.utils.data.DataLoader(trainset, **train_loader_kwargs)
        testloader = torch.utils.data.DataLoader(testset, **test_loader_kwargs)
        steps_per_epoch = len(trainloader)

    model = LeNet5(c1_channels=args.c1_channels).to(device)
    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile enabled")
        except Exception as exc:
            print(f"torch.compile failed, using eager mode: {exc}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=args.pct_start,
        div_factor=args.div_factor,
        final_div_factor=args.final_div_factor,
    )

    maybe_sync(device)
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0 if args.verbose else None
        if args.in_memory:
            full_train_size = steps_per_epoch * args.batch_size
            perm = torch.randperm(len(train_images), device=device)
            for offset in range(0, full_train_size, args.batch_size):
                idx = perm[offset : offset + args.batch_size]
                images = train_images[idx]
                labels = train_labels[idx]
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                if args.verbose:
                    running_loss += loss.item()
        else:
            for images, labels in trainloader:
                images = images.to(device, non_blocking=non_blocking)
                labels = labels.to(device, non_blocking=non_blocking)

                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                if args.verbose:
                    running_loss += loss.item()

        if args.verbose and running_loss is not None:
            avg_loss = running_loss / steps_per_epoch
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch + 1}/{args.epochs} "
                f"loss={avg_loss:.4f} lr={current_lr:.6f}"
            )

    maybe_sync(device)
    train_time = time.perf_counter() - start

    if args.in_memory:
        base_acc = evaluate_tensors(
            model, test_images, test_labels, args.batch_size * 2, tta_shifts=0
        )
    else:
        base_acc = evaluate(model, testloader, device, non_blocking, tta_shifts=0)
    if args.tta_shifts > 0:
        if args.in_memory:
            tta_acc = evaluate_tensors(
                model,
                test_images,
                test_labels,
                args.batch_size * 2,
                tta_shifts=args.tta_shifts,
            )
        else:
            tta_acc = evaluate(
                model, testloader, device, non_blocking, tta_shifts=args.tta_shifts
            )
        reported_acc = tta_acc
    else:
        tta_acc = None
        reported_acc = base_acc

    print("")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Train time: {train_time:.3f} s")
    print(f"Test accuracy (base): {base_acc:.3f} %")
    if tta_acc is not None:
        print(f"Test accuracy (TTA x{args.tta_shifts}): {tta_acc:.3f} %")
    print(f"Target: <{args.target_time:.1f}s and >{args.target_acc:.1f}%")
    print(
        "Beat benchmark: "
        f"{train_time < args.target_time and reported_acc > args.target_acc}"
    )


if __name__ == "__main__":
    main()
