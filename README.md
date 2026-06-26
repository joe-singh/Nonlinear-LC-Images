# Nonlinear LC Images

Toy experiments for image generation with passive nonlinear LC oscillator
networks coupled by varactor-like capacitances.

This repository is an experimental fork/scaffold based on
[unconv-ai/Un-0](https://github.com/unconv-ai/Un-0). The original Kuramoto
code path is kept intact for reference, but the new work here is the passive LC
toy baseline in `un0/passive_lc.py` and
`scripts/train_passive_lc_toy_cifar10.py`.

This is not an official Un-0 release and is not intended to reproduce Un-0
training quality. It is a small, Colab-friendly baseline for quickly testing
whether passive nonlinear LC dynamics are useful as an oscillator block in an
image-generation scaffold.

## Passive LC Toy Model

The LC state is:

```text
state = concat(phi, V)
```

where `phi` is node flux and `V = dphi/dt` is node voltage. The toy dynamics
are dimensionless:

```text
dphi/dt = V
M(V) dV/dt = -phi / L
```

The voltage-dependent mass matrix is:

```text
M(V) = diag(C_i) + sum_edges Cg_e(V_src - V_dst) b_e b_e^T
```

Each edge uses a varactor-like differential capacitance:

```text
v_safe = v_clip_scale * tanh((V_src - V_dst) / v_clip_scale)
Cg = Cj0 / (1 + (Vbias + v_safe) / Vj)^m
```

The model is an ideal lossless nonlinear differential-capacitance network. It
does not include damping, drive terms, diode conduction, or SPICE-level device
behavior.

## What Was Added

- `un0/passive_lc.py`: graph builder, passive LC dynamics, fixed-step Euler/RK4
  integrator, and a small class-conditional CIFAR-10 generator.
- `scripts/train_passive_lc_toy_cifar10.py`: lightweight toy training script
  using pixel-only conditional drift loss.
- `tests/test_passive_lc.py`: CPU tests that do not download CIFAR-10.
- `docs/passive_lc_toy.md`: experiment notes.
- `docs/passive_lc_toy_colab.md`: copy-paste Colab setup and run commands.

## Local Setup

Create a local environment and install the small toy requirements:

```bash
python -m venv .venv
.venv/bin/python -m pip install torch torchvision pytest tqdm torchdiffeq
```

Run the focused tests:

```bash
.venv/bin/python -m pytest tests/test_passive_lc.py -q
```

Run the no-download smoke test:

```bash
.venv/bin/python scripts/train_passive_lc_toy_cifar10.py \
    --synthetic-data \
    --epochs 1 \
    --subset-size 128 \
    --batch-size 32 \
    --n-oscillators 16 \
    --num-steps 2 \
    --max-batches 1 \
    --device cpu \
    --out-dir /tmp/passive_lc_smoke
```

## Colab Setup

In Colab, keep the existing PyTorch/CUDA install and install this repo without
dependency resolution:

```python
!git clone --branch passive-lc-toy https://github.com/joe-singh/Nonlinear-LC-Images.git
%cd Nonlinear-LC-Images
!pip install -q -e . --no-deps
!pip install -q torchdiffeq torchvision tqdm
```

Recommended first toy run:

```python
!python scripts/train_passive_lc_toy_cifar10.py \
    --device cuda \
    --precision fp16 \
    --n-oscillators 64 \
    --topology ring \
    --num-steps 8 \
    --method rk4 \
    --epochs 3 \
    --batch-size 256 \
    --subset-size 10000 \
    --out-dir runs/passive_lc_toy_colab
```

Sample grids are saved as `samples_epoch_*.png`; checkpoints are saved as
`latest.pt` and `final.pt` in the selected run directory.

## First Comparisons

Useful first checks:

- `--freeze-dynamics` versus learned dynamics.
- `--topology ring` versus `--topology random_sparse --k 4`.
- `--n-oscillators 32`, `64`, and `128`.
- `--num-steps 4` versus `8`.

## Upstream Attribution

This repo started from
[unconv-ai/Un-0](https://github.com/unconv-ai/Un-0), an MIT-licensed reference
implementation of Kuramoto-based image generation. Upstream files are retained
where useful so the original Kuramoto path can still be inspected and compared
against the passive LC toy baseline.
