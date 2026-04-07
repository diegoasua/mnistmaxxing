#!/usr/bin/env python3
import argparse
import os
import random
import time
from functools import partial

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from torchvision.datasets import MNIST


class TinyConv(nn.Module):
    def __init__(self, c1: int = 16, c2: int = 32, hidden: int = 128) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, c1, 5)
        self.conv2 = nn.Conv2d(c1, c2, 5)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(c2 * 4 * 4, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.pool(nn.relu(self.conv1(x)))
        x = self.pool(nn.relu(self.conv2(x)))
        x = x.reshape((x.shape[0], -1))
        x = nn.relu(self.fc1(x))
        return self.fc2(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast MNIST benchmark trainer (MLX).")
    parser.add_argument("--data-dir", type=str, default="~/.pytorch/MNIST_data/")
    parser.add_argument("--batch-size", type=int, default=1200)
    parser.add_argument("--eval-batch-size", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=4)

    parser.add_argument("--max-lr", type=float, default=0.028)
    parser.add_argument("--pct-start", type=float, default=0.24)
    parser.add_argument("--div-factor", type=float, default=18.0)
    parser.add_argument("--final-div-factor", type=float, default=1e5)
    parser.add_argument("--weight-decay", type=float, default=2e-5)
    parser.add_argument("--label-smoothing", type=float, default=0.01)

    parser.add_argument("--c1-channels", type=int, default=16)
    parser.add_argument("--c2-channels", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-time", type=float, default=10.4)
    parser.add_argument("--target-acc", type=float, default=99.3)
    parser.add_argument("--tta-shifts", type=int, choices=[0, 5, 9], default=0)

    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--include-compile-time", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


def preprocess_images(images: np.ndarray) -> np.ndarray:
    x = images.astype(np.float32) / 255.0
    x = (x - 0.1307) / 0.3081
    return x[..., None]


def load_data(data_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    root = os.path.expanduser(data_dir)
    train = MNIST(root, train=True, download=True)
    test = MNIST(root, train=False, download=True)

    x_train = preprocess_images(train.data.numpy())
    y_train = train.targets.numpy().astype(np.int32)
    x_test = preprocess_images(test.data.numpy())
    y_test = test.targets.numpy().astype(np.int32)
    return x_train, y_train, x_test, y_test


def build_one_cycle_schedule(
    total_steps: int,
    max_lr: float,
    pct_start: float,
    div_factor: float,
    final_div_factor: float,
):
    up_steps = max(1, int(total_steps * pct_start))
    down_steps = max(1, total_steps - up_steps)
    initial_lr = max_lr / div_factor
    final_lr = initial_lr / final_div_factor

    one = mx.array(1.0, dtype=mx.float32)
    zero = mx.array(0.0, dtype=mx.float32)
    max_step = mx.array(float(total_steps - 1), dtype=mx.float32)

    def schedule(step: mx.array) -> mx.array:
        s = mx.minimum(step.astype(mx.float32), max_step)
        up_ratio = mx.minimum(s / float(up_steps), one)
        down_ratio = mx.minimum(
            mx.maximum((s - float(up_steps)) / float(down_steps), zero), one
        )

        up_lr = initial_lr + (max_lr - initial_lr) * up_ratio
        down_cos = 0.5 * (1.0 + mx.cos(mx.pi * down_ratio))
        down_lr = final_lr + (max_lr - final_lr) * down_cos
        return mx.where(s <= float(up_steps), up_lr, down_lr)

    return schedule


def tta_shift_list(tta_shifts: int) -> list[tuple[int, int]]:
    if tta_shifts == 0:
        return [(0, 0)]
    if tta_shifts == 5:
        return [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    return [
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


def evaluate(
    model: nn.Module,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int,
    tta_shifts: int,
) -> float:
    shifts = tta_shift_list(tta_shifts)
    model.eval()
    correct = 0

    for i in range(0, len(x_test), batch_size):
        xb = mx.array(x_test[i : i + batch_size])
        yb = mx.array(y_test[i : i + batch_size])

        if len(shifts) == 1:
            logits = model(xb)
        else:
            logits = None
            for dy, dx in shifts:
                x_roll = mx.roll(xb, shift=(dy, dx), axis=(1, 2))
                out = model(x_roll)
                logits = out if logits is None else logits + out
            logits = logits / float(len(shifts))

        pred = mx.argmax(logits, axis=-1)
        correct += int(mx.sum(pred == yb).item())

    return 100.0 * correct / len(x_test)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    x_train, y_train, x_test, y_test = load_data(args.data_dir)

    train_steps_per_epoch = len(x_train) // args.batch_size
    if train_steps_per_epoch < 1:
        raise ValueError("batch-size is larger than training set")
    total_steps = train_steps_per_epoch * args.epochs

    schedule = build_one_cycle_schedule(
        total_steps=total_steps,
        max_lr=args.max_lr,
        pct_start=args.pct_start,
        div_factor=args.div_factor,
        final_div_factor=args.final_div_factor,
    )

    model = TinyConv(
        c1=args.c1_channels,
        c2=args.c2_channels,
        hidden=args.hidden,
    )
    optimizer = optim.AdamW(
        learning_rate=schedule,
        weight_decay=args.weight_decay,
        bias_correction=True,
    )

    def loss_fn(xb: mx.array, yb: mx.array) -> mx.array:
        logits = model(xb)
        return nn.losses.cross_entropy(
            logits,
            yb,
            label_smoothing=args.label_smoothing,
            reduction="mean",
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def eager_step(xb: mx.array, yb: mx.array) -> mx.array:
        loss, grads = loss_and_grad(xb, yb)
        optimizer.update(model, grads)
        return loss

    if args.no_compile:
        train_step = eager_step
    else:
        train_step = partial(
            mx.compile,
            inputs=[model.state, optimizer.state],
            outputs=[model.state, optimizer.state],
        )(eager_step)

    full_train_size = train_steps_per_epoch * args.batch_size

    if not args.no_compile and not args.include_compile_time:
        warm_x = mx.array(x_train[: args.batch_size])
        warm_y = mx.array(y_train[: args.batch_size])
        warm_loss = train_step(warm_x, warm_y)
        mx.eval(warm_loss)

    model.train()
    start = time.perf_counter()
    for epoch in range(args.epochs):
        perm = np.random.permutation(len(x_train))
        running_loss = 0.0 if args.verbose else None

        for offset in range(0, full_train_size, args.batch_size):
            idx = perm[offset : offset + args.batch_size]
            xb = mx.array(x_train[idx])
            yb = mx.array(y_train[idx])
            loss = train_step(xb, yb)
            mx.eval(loss)
            if args.verbose and running_loss is not None:
                running_loss += float(loss.item())

        if args.verbose and running_loss is not None:
            avg_loss = running_loss / train_steps_per_epoch
            print(f"Epoch {epoch + 1}/{args.epochs} loss={avg_loss:.4f}")

    mx.eval(model.state, optimizer.state)
    train_time = time.perf_counter() - start

    base_acc = evaluate(
        model,
        x_test,
        y_test,
        batch_size=args.eval_batch_size,
        tta_shifts=0,
    )
    if args.tta_shifts > 0:
        tta_acc = evaluate(
            model,
            x_test,
            y_test,
            batch_size=args.eval_batch_size,
            tta_shifts=args.tta_shifts,
        )
        reported_acc = tta_acc
    else:
        tta_acc = None
        reported_acc = base_acc

    print("")
    print(f"Device: {mx.default_device()}")
    print(f"MLX: {mx.__version__}")
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
