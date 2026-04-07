Use either:

- `main.ipynb` (original notebook)
- `main.py` (clean PyTorch script, tuned for speed on Apple Silicon)
- `main_mlx.py` (experimental MLX script)

## Reproducible Fast Run (MacBook Air M2)

```bash
python3.14 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install torch torchvision
python main.py
```

On this machine (`python 3.14.3`, `torch 2.11.0`, MPS), `main.py` defaults produced:

- train time: `~5.37s`
- test accuracy: `~99.31%`

This beats the `10.4s @ 99.3%` benchmark.

Use `python main.py --verbose` if you want per-epoch logs.
By default, `main.py` uses `--in-memory` to keep MNIST tensors on-device and reduce dataloader overhead.
Use `python main.py --no-in-memory` if you want the streaming loader path.

## Optional Aggressive TTA Mode

If you allow test-time augmentation (still no extra train time), run:

```bash
python main.py --tta-shifts 5
```

Observed on this machine:

- train time: `~5.4s`
- base accuracy: `~99.31%`
- TTA(5-shift) accuracy: `~99.36%`

## Optional Higher Accuracy Mode

```bash
python main.py --epochs 5 --lr 0.024
```

Observed on this machine:

- train time: `~6.7-7.3s` (machine/thermal dependent)
- base accuracy: `~99.34-99.45%`
