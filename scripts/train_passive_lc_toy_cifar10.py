"""Train a small passive nonlinear LC toy generator on CIFAR-10 pixels."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from un0.losses import conditional_drift_loss_for_views  # noqa: E402
from un0.passive_lc import PassiveLCGenerator  # noqa: E402

IMAGE_SIZE = 32
NUM_CLASSES = 10
GRAD_CLIP_NORM = 1.0
LC_PARAMETER_NAMES = ("C", "L", "Cj0", "Vbias")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-oscillators", type=int, default=64)
    parser.add_argument(
        "--topology",
        choices=("ring", "chain", "random_sparse", "all_to_all"),
        default="ring",
    )
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=8)
    parser.add_argument("--method", choices=("euler", "rk4"), default="rk4")
    parser.add_argument("--integration-time", type=float, default=1.0)
    parser.add_argument(
        "--decoder-type",
        choices=("conv", "linear"),
        default="conv",
        help="Use the resize-conv image decoder or a direct linear pixel decoder.",
    )
    parser.add_argument(
        "--decoder-width",
        type=int,
        default=32,
        help="Hidden channel width for the toy resize-conv decoder.",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--dynamics-lr-multiplier",
        type=float,
        default=1.0,
        help="Multiply the base learning rate for trainable LC dynamics parameters.",
    )
    parser.add_argument("--subset-size", type=int, default=10000)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp32")
    parser.add_argument("--out-dir", default="runs/passive_lc_toy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--freeze-dynamics",
        action="store_true",
        help="Freeze LC dynamics parameters and train the rest as a reservoir baseline.",
    )
    parser.add_argument(
        "--label-only",
        action="store_true",
        help="Decode learned class embeddings only; no random LC state or dynamics rollout.",
    )
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument(
        "--synthetic-data",
        action="store_true",
        help="Use random tensors instead of downloading CIFAR-10, for smoke tests.",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--num-workers", type=int, default=2)
    return parser


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32":
        return nullcontext()
    if device.type == "cuda":
        dtype = torch.float16 if precision == "fp16" else torch.bfloat16
        return torch.amp.autocast(device_type="cuda", dtype=dtype)
    if device.type == "cpu" and precision == "bf16":
        return torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)
    return nullcontext()


def build_loader(args: argparse.Namespace, device: torch.device) -> DataLoader:
    subset_size = int(args.subset_size)
    if args.synthetic_data:
        num_samples = subset_size if subset_size > 0 else 1024
        generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
        images = torch.rand(
            num_samples,
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
            generator=generator,
        )
        images = images * 2.0 - 1.0
        labels = torch.arange(num_samples, dtype=torch.long) % NUM_CLASSES
        dataset = TensorDataset(images, labels)
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = datasets.CIFAR10(
            root=str(args.data_dir),
            train=True,
            download=True,
            transform=transform,
        )
        if 0 < subset_size < len(dataset):
            generator = torch.Generator(device="cpu").manual_seed(int(args.seed))
            indices = torch.randperm(len(dataset), generator=generator)[:subset_size].tolist()
            dataset = Subset(dataset, indices)

    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def class_balanced_labels(
    *,
    samples_per_class: int = 8,
    device: torch.device,
) -> Tensor:
    return torch.arange(NUM_CLASSES, device=device).repeat_interleave(samples_per_class)


@torch.no_grad()
def save_samples(model: PassiveLCGenerator, out_dir: Path, epoch: int, device: torch.device) -> None:
    labels = class_balanced_labels(samples_per_class=8, device=device)
    images = model.sample_images(labels)
    save_image(images.cpu(), out_dir / f"samples_epoch_{epoch:03d}.png", nrow=8)


def build_model(args: argparse.Namespace, device: torch.device) -> PassiveLCGenerator:
    model = PassiveLCGenerator(
        n_oscillators=int(args.n_oscillators),
        topology=args.topology,
        k=int(args.k),
        seed=int(args.seed),
        num_steps=int(args.num_steps),
        integration_time=float(args.integration_time),
        method=args.method,
        num_classes=NUM_CLASSES,
        label_only=bool(args.label_only),
        decoder_width=int(args.decoder_width),
        decoder_type=args.decoder_type,
    ).to(device)
    if args.freeze_dynamics or args.label_only:
        for parameter in model.dynamics.parameters():
            parameter.requires_grad_(False)
    return model


@torch.no_grad()
def lc_parameter_snapshot(model: PassiveLCGenerator) -> dict[str, Tensor]:
    """Return detached physical LC parameters, not raw unconstrained tensors."""
    return {
        name: value.detach().float().cpu().clone()
        for name, value in zip(LC_PARAMETER_NAMES, model.dynamics.positive_parameters())
    }


def tensor_summary(tensor: Tensor) -> dict[str, float]:
    values = tensor.detach().float().cpu()
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def lc_parameter_delta_summary(
    model: PassiveLCGenerator,
    reference: dict[str, Tensor],
) -> dict[str, dict[str, float]]:
    """Summarize absolute movement in physical LC parameters since reference."""
    current = lc_parameter_snapshot(model)
    summary: dict[str, dict[str, float]] = {}
    for name in LC_PARAMETER_NAMES:
        delta = (current[name] - reference[name]).abs()
        relative = delta / reference[name].abs().clamp_min(1e-12)
        summary[name] = {
            "mean_abs": float(delta.mean()),
            "max_abs": float(delta.max()),
            "mean_relative": float(relative.mean()),
        }
    return summary


def _parameter_grad_norm(parameters) -> float:
    total_sq = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        if not torch.isfinite(grad).all():
            return float("nan")
        norm = float(grad.norm(2))
        total_sq += norm * norm
    return total_sq**0.5


def module_gradient_norms(model: PassiveLCGenerator) -> dict[str, float]:
    """Return L2 gradient norms for the main trainable module groups."""
    readout_parameters = list(model.readout_norm.parameters()) + list(model.readout.parameters())
    return {
        "dynamics": _parameter_grad_norm(model.dynamics.parameters()),
        "class_offset": _parameter_grad_norm(model.class_offset.parameters()),
        "readout": _parameter_grad_norm(readout_parameters),
        "decoder": _parameter_grad_norm(model.decoder.parameters()),
    }


def build_optimizer(
    model: PassiveLCGenerator,
    *,
    lr: float,
    dynamics_lr_multiplier: float,
) -> tuple[torch.optim.Optimizer, list[nn.Parameter], dict[str, float]]:
    """Build AdamW with an optional higher LR for trainable LC parameters."""
    if lr <= 0.0:
        raise ValueError(f"lr must be positive, got {lr}.")
    if dynamics_lr_multiplier <= 0.0:
        raise ValueError(
            f"dynamics_lr_multiplier must be positive, got {dynamics_lr_multiplier}."
        )

    dynamics_parameters = [
        parameter for parameter in model.dynamics.parameters() if parameter.requires_grad
    ]
    dynamics_parameter_ids = {id(parameter) for parameter in dynamics_parameters}
    other_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in dynamics_parameter_ids
    ]
    trainable = dynamics_parameters + other_parameters
    if not trainable:
        raise ValueError("No trainable parameters remain after applying --freeze-dynamics.")

    dynamics_lr = lr * dynamics_lr_multiplier
    parameter_groups = []
    if dynamics_parameters:
        parameter_groups.append(
            {
                "params": dynamics_parameters,
                "lr": dynamics_lr,
                "name": "dynamics",
            }
        )
    if other_parameters:
        parameter_groups.append(
            {
                "params": other_parameters,
                "lr": lr,
                "name": "non_dynamics",
            }
        )

    optimizer = torch.optim.AdamW(parameter_groups)
    optimizer_lrs = {
        "dynamics": dynamics_lr if dynamics_parameters else 0.0,
        "non_dynamics": lr if other_parameters else 0.0,
    }
    return optimizer, trainable, optimizer_lrs


def write_diagnostics(out_dir: Path, diagnostics: dict[str, Any]) -> None:
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n")


def format_grad_norms(norms: dict[str, float]) -> str:
    return (
        "grad_norm "
        f"dyn={norms['dynamics']:.2e} "
        f"readout={norms['readout']:.2e} "
        f"decoder={norms['decoder']:.2e}"
    )


def format_lc_delta(delta: dict[str, dict[str, float]]) -> str:
    return (
        "lc_delta "
        f"C={delta['C']['mean_abs']:.2e} "
        f"L={delta['L']['mean_abs']:.2e} "
        f"Cj0={delta['Cj0']['mean_abs']:.2e} "
        f"Vbias={delta['Vbias']['mean_abs']:.2e}"
    )


def checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    if diagnostics is not None:
        payload["diagnostics"] = diagnostics
    return payload


def train(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = build_loader(args, device)
    model = build_model(args, device)
    optimizer, trainable, optimizer_lrs = build_optimizer(
        model,
        lr=float(args.lr),
        dynamics_lr_multiplier=float(args.dynamics_lr_multiplier),
    )
    initial_lc_parameters = lc_parameter_snapshot(model)
    diagnostics: dict[str, Any] = {
        "mode": None,
        "args": vars(args),
        "optimizer_lrs": optimizer_lrs,
        "initial_lc_parameters": {
            name: tensor_summary(value) for name, value in initial_lc_parameters.items()
        },
        "epochs": [],
    }
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "fp16")

    if args.label_only:
        mode = "label_only"
    elif args.freeze_dynamics:
        mode = "frozen_lc"
    else:
        mode = "learned_lc"
    diagnostics["mode"] = mode
    write_diagnostics(out_dir, diagnostics)
    print(
        f"device={device} precision={args.precision} mode={mode} "
        f"decoder_type={args.decoder_type} decoder_width={args.decoder_width} "
        f"lr={float(args.lr):.2e} dynamics_lr={optimizer_lrs['dynamics']:.2e} "
        f"batches_per_epoch={len(loader)}"
    )
    save_samples(model, out_dir, epoch=0, device=device)

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        grad_norm_totals = {
            "dynamics": 0.0,
            "class_offset": 0.0,
            "readout": 0.0,
            "decoder": 0.0,
        }
        progress = tqdm(loader, desc=f"epoch {epoch}", leave=False)
        for images, labels in progress:
            if int(args.max_batches) > 0 and num_batches >= int(args.max_batches):
                break

            labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
            real_flat = images.to(device=device, non_blocking=True).reshape(labels.shape[0], -1)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.precision):
                gen_flat = model(labels)

            # The pixel drift target uses high-dimensional pairwise distances
            # and a large self-mask constant. Keep it in fp32 even when the
            # generator forward runs under fp16/bf16 autocast.
            loss = conditional_drift_loss_for_views(
                [(real_flat.float(), gen_flat.float())],
                class_id_pos=labels,
                class_id_gen=labels,
                gamma=0.2,
                compile_drift=False,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}, batch {num_batches}: "
                    f"{float(loss.detach().cpu())}."
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            batch_grad_norms = module_gradient_norms(model)
            for name, value in batch_grad_norms.items():
                grad_norm_totals[name] += value
            torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            loss_value = float(loss.detach().cpu())
            total_loss += loss_value
            num_batches += 1
            progress.set_postfix(loss=f"{loss_value:.4f}")

        mean_loss = total_loss / max(num_batches, 1)
        mean_grad_norms = {
            name: value / max(num_batches, 1) for name, value in grad_norm_totals.items()
        }
        lc_delta = lc_parameter_delta_summary(model, initial_lc_parameters)
        diagnostics["epochs"].append(
            {
                "epoch": int(epoch),
                "loss": mean_loss,
                "mean_grad_norm": mean_grad_norms,
                "lc_parameter_delta_from_init": lc_delta,
            }
        )
        diagnostics["latest_lc_parameter_delta_from_init"] = lc_delta
        write_diagnostics(out_dir, diagnostics)
        print(
            f"epoch {epoch}: loss={mean_loss:.6f} "
            f"{format_grad_norms(mean_grad_norms)} {format_lc_delta(lc_delta)}"
        )

        torch.save(
            checkpoint_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                args=args,
                diagnostics=diagnostics,
            ),
            out_dir / "latest.pt",
        )
        if int(args.sample_every) > 0 and epoch % int(args.sample_every) == 0:
            save_samples(model, out_dir, epoch=epoch, device=device)

    torch.save(
        checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=int(args.epochs),
            args=args,
            diagnostics=diagnostics,
        ),
        out_dir / "final.pt",
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
