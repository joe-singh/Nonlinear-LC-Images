"""Compute clean-FID for passive LC toy checkpoints on CIFAR-10."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from un0.metrics import compute_fid  # noqa: E402
from un0.passive_lc import PassiveLCGenerator  # noqa: E402

IMAGE_SIZE = 32
NUM_CLASSES = 10
DEFAULT_NUM_SAMPLES = 5000
DEFAULT_BATCH_SIZE = 256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--num-samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=(
            "Generated samples to score. Use 5000-10000 for quick model ranking; "
            "50000 for a slower, more standard CIFAR-10 FID."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Optional directory to keep generated PNGs. Defaults to a tempdir.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional JSON path for {fid, checkpoint, num_samples, seed, config}.",
    )
    return parser


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _checkpoint_args(state: dict[str, Any]) -> dict[str, Any]:
    args = state.get("args", {})
    if not isinstance(args, dict):
        raise ValueError("Checkpoint is missing a dict-like 'args' payload.")
    return args


def build_model_from_checkpoint_args(
    ckpt_args: dict[str, Any],
    *,
    device: torch.device,
) -> PassiveLCGenerator:
    """Rebuild the toy generator from the training script's saved CLI args."""
    model = PassiveLCGenerator(
        n_oscillators=int(ckpt_args.get("n_oscillators", 64)),
        topology=str(ckpt_args.get("topology", "ring")),
        k=int(ckpt_args.get("k", 4)),
        seed=int(ckpt_args.get("seed", 42)),
        num_steps=int(ckpt_args.get("num_steps", 8)),
        integration_time=float(ckpt_args.get("integration_time", 1.0)),
        method=str(ckpt_args.get("method", "rk4")),
        num_classes=NUM_CLASSES,
        label_only=bool(ckpt_args.get("label_only", False)),
        decoder_width=int(ckpt_args.get("decoder_width", 32)),
    )
    return model.to(device)


def evaluate(args: argparse.Namespace) -> float:
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    device = resolve_device(args.device)

    try:
        from un0.common import disable_torchscript_gpu_fuser_on_blackwell

        disable_torchscript_gpu_fuser_on_blackwell()
    except Exception:
        # Only needed for specific Blackwell/TorchScript clean-FID crashes.
        pass

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    ckpt_args = _checkpoint_args(state)
    model = build_model_from_checkpoint_args(ckpt_args, device=device)
    model.load_state_dict(state["model"])
    model.eval()

    fid = compute_fid(
        model,
        num_samples=int(args.num_samples),
        num_classes=NUM_CLASSES,
        batch_size=int(args.batch_size),
        device=device,
        image_size=IMAGE_SIZE,
        image_dir=args.image_dir,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "fid": fid,
                    "checkpoint": str(args.checkpoint),
                    "num_samples": int(args.num_samples),
                    "batch_size": int(args.batch_size),
                    "seed": int(args.seed),
                    "config": ckpt_args,
                },
                indent=2,
            )
        )
    return fid


def main() -> None:
    args = build_parser().parse_args()
    fid = evaluate(args)
    print(f"FID: {fid:.4f}")


if __name__ == "__main__":
    main()
