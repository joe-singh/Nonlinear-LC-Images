"""Train a small passive nonlinear LC toy generator on CIFAR-10 pixels."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
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
    ).to(device)
    if args.freeze_dynamics:
        for parameter in model.dynamics.parameters():
            parameter.requires_grad_(False)
    return model


def checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "epoch": int(epoch),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }


def train(args: argparse.Namespace) -> None:
    seed_everything(int(args.seed))
    device = resolve_device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = build_loader(args, device)
    model = build_model(args, device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters remain after applying --freeze-dynamics.")
    optimizer = torch.optim.AdamW(trainable, lr=float(args.lr))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "fp16")

    print(f"device={device} precision={args.precision} batches_per_epoch={len(loader)}")
    save_samples(model, out_dir, epoch=0, device=device)

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0
        num_batches = 0
        progress = tqdm(loader, desc=f"epoch {epoch}", leave=False)
        for images, labels in progress:
            if int(args.max_batches) > 0 and num_batches >= int(args.max_batches):
                break

            labels = labels.to(device=device, dtype=torch.long, non_blocking=True)
            real_flat = images.to(device=device, non_blocking=True).reshape(labels.shape[0], -1)

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.precision):
                gen_flat = model(labels)
                loss = conditional_drift_loss_for_views(
                    [(real_flat, gen_flat)],
                    class_id_pos=labels,
                    class_id_gen=labels,
                    gamma=0.2,
                    compile_drift=False,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()

            loss_value = float(loss.detach().cpu())
            total_loss += loss_value
            num_batches += 1
            progress.set_postfix(loss=f"{loss_value:.4f}")

        mean_loss = total_loss / max(num_batches, 1)
        print(f"epoch {epoch}: loss={mean_loss:.6f}")

        torch.save(
            checkpoint_payload(model=model, optimizer=optimizer, epoch=epoch, args=args),
            out_dir / "latest.pt",
        )
        if int(args.sample_every) > 0 and epoch % int(args.sample_every) == 0:
            save_samples(model, out_dir, epoch=epoch, device=device)

    torch.save(
        checkpoint_payload(model=model, optimizer=optimizer, epoch=int(args.epochs), args=args),
        out_dir / "final.pt",
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
