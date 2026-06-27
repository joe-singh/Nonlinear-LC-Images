# Passive LC Toy Colab Notes

These cells keep Colab's existing PyTorch/CUDA install and install this repo in
editable mode without dependency resolution.

## Setup

```python
!git clone --branch passive-lc-toy https://github.com/joe-singh/Nonlinear-LC-Images.git
%cd Nonlinear-LC-Images
```

```python
# Keep Colab's existing torch install.
!pip install -q -e . --no-deps
!pip install -q torchdiffeq torchvision tqdm
```

## Recommended Toy Run

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

## Smaller Sanity Run

```python
!python scripts/train_passive_lc_toy_cifar10.py \
    --device cuda \
    --precision fp16 \
    --n-oscillators 32 \
    --num-steps 4 \
    --epochs 1 \
    --batch-size 128 \
    --subset-size 1024 \
    --max-batches 5 \
    --out-dir runs/passive_lc_smoke
```

## Outputs

Sample grids are saved in the chosen run directory:

```text
runs/passive_lc_toy_colab/samples_epoch_000.png
runs/passive_lc_toy_colab/samples_epoch_001.png
...
```

Checkpoints are saved as:

```text
runs/passive_lc_toy_colab/latest.pt
runs/passive_lc_toy_colab/final.pt
```

Training diagnostics are saved as:

```text
runs/passive_lc_toy_colab/diagnostics.json
```

Quickly inspect the final gradient norms and LC parameter movement:

```python
import json

with open("runs/passive_lc_toy_colab/diagnostics.json") as f:
    diagnostics = json.load(f)

last = diagnostics["epochs"][-1]
print("grad norms:", last["mean_grad_norm"])
print("LC delta:", last["lc_parameter_delta_from_init"])
```

If learned LC has much smaller dynamics gradients than decoder/readout, try a
dynamics learning-rate boost:

```python
!python scripts/train_passive_lc_toy_cifar10.py \
    --device cuda \
    --precision bf16 \
    --seed 42 \
    --decoder-width 8 \
    --n-oscillators 64 \
    --topology ring \
    --num-steps 8 \
    --method rk4 \
    --dynamics-lr-multiplier 10 \
    --epochs 25 \
    --batch-size 256 \
    --subset-size 10000 \
    --out-dir runs/weakdec_learned_lc_n64_w8_dynlr10
```

## Linear Decoder Ablation

Use `--decoder-type linear` to replace the resize-conv stack with a direct
`LayerNorm -> Linear -> tanh` pixel head. Run these three cells and compare FID:

```python
# Decoder-only linear head
!python scripts/train_passive_lc_toy_cifar10.py \
    --device cuda \
    --precision bf16 \
    --seed 42 \
    --decoder-type linear \
    --n-oscillators 64 \
    --topology ring \
    --num-steps 0 \
    --method rk4 \
    --epochs 25 \
    --batch-size 256 \
    --subset-size 10000 \
    --out-dir runs/linear_decoder_only_n64
```

```python
# Frozen LC with linear head
!python scripts/train_passive_lc_toy_cifar10.py \
    --device cuda \
    --precision bf16 \
    --seed 42 \
    --decoder-type linear \
    --n-oscillators 64 \
    --topology ring \
    --num-steps 8 \
    --method rk4 \
    --freeze-dynamics \
    --epochs 25 \
    --batch-size 256 \
    --subset-size 10000 \
    --out-dir runs/linear_frozen_lc_n64
```

```python
# Learned LC with linear head
!python scripts/train_passive_lc_toy_cifar10.py \
    --device cuda \
    --precision bf16 \
    --seed 42 \
    --decoder-type linear \
    --n-oscillators 64 \
    --topology ring \
    --num-steps 8 \
    --method rk4 \
    --epochs 25 \
    --batch-size 256 \
    --subset-size 10000 \
    --out-dir runs/linear_learned_lc_n64
```

## FID Scoring

Install clean-FID only when you want to score checkpoints:

```python
!pip install -q clean-fid
```

Quick 5k-sample ranking:

```python
!python scripts/eval_passive_lc_toy_fid.py \
    --checkpoint runs/passive_lc_toy_colab/final.pt \
    --num-samples 5000 \
    --batch-size 256 \
    --device cuda \
    --output runs/passive_lc_toy_colab/fid_5k.json
```

Use `--num-samples 50000` for a slower, more standard CIFAR-10 clean-FID score.

To pull updates after changes are pushed:

```python
!git pull
!pip install -q -e . --no-deps
```
