# Passive Nonlinear LC Toy Baseline

This experiment adds a small passive nonlinear LC replacement for the
Kuramoto dynamics block used by Un-0. It is a training and debugging scaffold,
not a full-quality image generator and not a SPICE-level circuit simulator.

## Model

The LC state is `concat(phi, V)`, where `phi` is node flux and `V = dphi/dt`
is node voltage. The prompt-compatible dimensionless dynamics are:

```text
dphi/dt = V
M(V) dV/dt = -phi / L
```

The mass matrix contains positive node capacitances plus nonlinear coupling
capacitances on graph edges:

```text
M(V) = diag(C_i) + sum_edges Cg_e(V_src - V_dst) b_e b_e^T
```

Each edge uses a varactor-like differential capacitance:

```text
v_safe = v_clip_scale * tanh((V_src - V_dst) / v_clip_scale)
Cg = Cj0 / (1 + (Vbias + v_safe) / Vj)^m
```

This is an ideal lossless nonlinear differential-capacitance network. It does
not include damping, drive terms, diode conduction, parasitics beyond `C_i`, or
device-calibrated varactor behavior.

## Generator

`PassiveLCGenerator` samples random initial LC states, adds a learned
class-dependent initial-state offset, rolls out the LC dynamics for a small
fixed number of Euler or RK4 steps, and decodes the final state into CIFAR-10
images.

The default toy readout is:

```text
LayerNorm(final_state) -> Linear(2N, 128) -> tanh -> resize-conv decoder
```

The output is a flat tensor with shape `(batch, 3 * 32 * 32)` in `[-1, 1]`.

## Training

The toy script trains class-conditionally on CIFAR-10 using only the pixel
version of Un-0's conditional drift loss:

```bash
python scripts/train_passive_lc_toy_cifar10.py \
    --device cuda \
    --precision fp16 \
    --n-oscillators 64 \
    --topology ring \
    --num-steps 8 \
    --method rk4 \
    --epochs 3 \
    --batch-size 256 \
    --subset-size 10000 \
    --out-dir runs/passive_lc_toy
```

No DINO, FID, W&B, DDP, or `torch.compile` is used in this path.

For a no-download CPU smoke test:

```bash
python scripts/train_passive_lc_toy_cifar10.py \
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

Sample grids are saved as `samples_epoch_*.png`; checkpoints are saved as
`latest.pt` and `final.pt`.

## FID Scoring

The toy path does not compute FID during training, but checkpoints can be scored
afterward with clean-FID:

```bash
pip install clean-fid

python scripts/eval_passive_lc_toy_fid.py \
    --checkpoint runs/weakdec_learned_lc_n64_w8/final.pt \
    --num-samples 5000 \
    --batch-size 256 \
    --device cuda \
    --output runs/weakdec_learned_lc_n64_w8/fid_5k.json
```

Use `--num-samples 5000` or `10000` for quick ranking across ablations. Use
`--num-samples 50000` for a slower, more standard CIFAR-10 clean-FID estimate.
Small-sample FID is noisy, but it is still useful for deciding which toy
configuration deserves a longer run.

## First Comparisons

Good first runs are:

- `--label-only` to test whether the decoder can learn class prototypes
  without any random or LC latent state.
- `--freeze-dynamics` versus learned dynamics.
- `--num-steps 0` to test a decoder-only random latent baseline.
- `--decoder-width 8` or `16` to reduce decoder capacity and expose whether
  the LC rollout contributes useful latent structure.
- `--topology ring` versus `--topology random_sparse --k 4`.
- `--n-oscillators 32`, `64`, and `128`.
- `--num-steps 4` versus `8`.

For a first dynamics-matter ablation, keep the seed, subset, decoder, and
training budget fixed:

```bash
python scripts/train_passive_lc_toy_cifar10.py \
    --device cuda --precision bf16 --seed 42 \
    --label-only \
    --n-oscillators 64 --epochs 25 --batch-size 256 --subset-size 10000 \
    --out-dir runs/ablate_label_only_n64

python scripts/train_passive_lc_toy_cifar10.py \
    --device cuda --precision bf16 --seed 42 \
    --n-oscillators 64 --num-steps 0 \
    --epochs 25 --batch-size 256 --subset-size 10000 \
    --out-dir runs/ablate_decoder_only_n64

python scripts/train_passive_lc_toy_cifar10.py \
    --device cuda --precision bf16 --seed 42 \
    --n-oscillators 64 --topology ring --num-steps 8 --method rk4 \
    --freeze-dynamics \
    --epochs 25 --batch-size 256 --subset-size 10000 \
    --out-dir runs/ablate_frozen_lc_n64

python scripts/train_passive_lc_toy_cifar10.py \
    --device cuda --precision bf16 --seed 42 \
    --n-oscillators 64 --topology ring --num-steps 8 --method rk4 \
    --epochs 25 --batch-size 256 --subset-size 10000 \
    --out-dir runs/ablate_learned_lc_n64
```

If those look similar, rerun the latent-bearing rows with a weaker decoder:

```bash
python scripts/train_passive_lc_toy_cifar10.py \
    --device cuda --precision bf16 --seed 42 \
    --decoder-width 8 \
    --n-oscillators 64 --num-steps 0 \
    --epochs 25 --batch-size 256 --subset-size 10000 \
    --out-dir runs/weakdec_decoder_only_n64_w8

python scripts/train_passive_lc_toy_cifar10.py \
    --device cuda --precision bf16 --seed 42 \
    --decoder-width 8 \
    --n-oscillators 64 --topology ring --num-steps 8 --method rk4 \
    --freeze-dynamics \
    --epochs 25 --batch-size 256 --subset-size 10000 \
    --out-dir runs/weakdec_frozen_lc_n64_w8

python scripts/train_passive_lc_toy_cifar10.py \
    --device cuda --precision bf16 --seed 42 \
    --decoder-width 8 \
    --n-oscillators 64 --topology ring --num-steps 8 --method rk4 \
    --epochs 25 --batch-size 256 --subset-size 10000 \
    --out-dir runs/weakdec_learned_lc_n64_w8
```

Do not expect this toy baseline to match released Un-0 sample quality. The
goal is to quickly test whether a passive nonlinear LC reservoir can serve as a
useful oscillator block in the image-generation scaffold.
