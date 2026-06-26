"""Passive nonlinear LC toy generator for CIFAR-sized images."""

from __future__ import annotations

from contextlib import nullcontext
import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

Topology = Literal["ring", "chain", "random_sparse", "all_to_all"]
IntegrationMethod = Literal["euler", "rk4"]


def _inv_softplus(value: float) -> float:
    if value <= 0.0:
        raise ValueError(f"value must be positive, got {value}.")
    return math.log(math.expm1(value))


def _logit(value: float) -> float:
    value = min(max(value, 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def make_lc_edges(
    n: int,
    topology: Topology = "ring",
    k: int = 4,
    seed: int = 0,
) -> torch.LongTensor:
    """Build undirected LC coupling edges as a ``(2, num_edges)`` tensor.

    The returned orientation is arbitrary and is used only to form voltage
    differences. ``all_to_all`` creates ``n * (n - 1) / 2`` edges and is only
    practical for small oscillator counts.
    """
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}.")
    if topology not in ("ring", "chain", "random_sparse", "all_to_all"):
        raise ValueError(f"Unsupported topology: {topology!r}.")

    if topology == "ring":
        pairs = [(i, (i + 1) % n) for i in range(n)]
    elif topology == "chain":
        pairs = [(i, i + 1) for i in range(n - 1)]
    elif topology == "all_to_all":
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        if k < 1:
            raise ValueError(f"k must be positive for random_sparse, got {k}.")
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        max_edges = len(all_pairs)
        num_edges = min(max_edges, max(1, round(n * k / 2)))
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        perm = torch.randperm(max_edges, generator=generator)[:num_edges].tolist()
        pairs = [all_pairs[i] for i in perm]

    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


class PassiveLCDynamics(nn.Module):
    """Dimensionless passive nonlinear LC dynamics.

    The state is ``concat(phi, V)`` with both components shaped ``(batch, n)``.
    The equations are the prompt-compatible ideal nonlinear differential
    capacitance model:

    ``dphi/dt = V``

    ``M(V) dV/dt = -phi / L``

    where ``M`` contains positive node capacitances plus edge varactor
    differential capacitances.
    """

    def __init__(
        self,
        *,
        n: int,
        edges: Tensor | None = None,
        topology: Topology = "ring",
        k: int = 4,
        seed: int = 0,
        init_c: float = 1.0,
        init_l: float = 1.0,
        init_cg: float = 0.05,
        init_vbias: float = 0.1,
        c_min: float = 1e-3,
        l_min: float = 1e-3,
        cg_min: float = 1e-4,
        cg_scale: float = 0.12,
        vbias_min: float = 2.05,
        vj: float = 1.0,
        m: float = 0.5,
        v_clip_scale: float = 2.0,
        jitter: float = 1e-5,
    ) -> None:
        super().__init__()
        if n < 2:
            raise ValueError(f"n must be at least 2, got {n}.")
        if vj <= 0.0:
            raise ValueError(f"vj must be positive, got {vj}.")
        if m <= 0.0:
            raise ValueError(f"m must be positive, got {m}.")
        if v_clip_scale <= 0.0:
            raise ValueError(f"v_clip_scale must be positive, got {v_clip_scale}.")
        if jitter <= 0.0:
            raise ValueError(f"jitter must be positive, got {jitter}.")

        if edges is None:
            edges = make_lc_edges(n, topology=topology, k=k, seed=seed)
        edges = torch.as_tensor(edges, dtype=torch.long)
        if edges.ndim != 2 or edges.shape[0] != 2 or edges.shape[1] == 0:
            raise ValueError("edges must have shape (2, num_edges) with at least one edge.")
        if int(edges.min()) < 0 or int(edges.max()) >= n:
            raise ValueError(f"edges contain node indices outside [0, {n}).")
        if torch.any(edges[0] == edges[1]):
            raise ValueError("self edges are not allowed.")

        self.n = int(n)
        self.topology = topology
        self.k = int(k)
        self.seed = int(seed)
        self.c_min = float(c_min)
        self.l_min = float(l_min)
        self.cg_min = float(cg_min)
        self.cg_scale = float(cg_scale)
        self.vbias_min = float(vbias_min)
        self.vj = float(vj)
        self.m = float(m)
        self.v_clip_scale = float(v_clip_scale)
        self.jitter = float(jitter)

        self.register_buffer("edges", edges.contiguous())
        incidence = torch.zeros(edges.shape[1], n, dtype=torch.float32)
        incidence[torch.arange(edges.shape[1]), edges[0]] = 1.0
        incidence[torch.arange(edges.shape[1]), edges[1]] = -1.0
        self.register_buffer("incidence", incidence)

        self.raw_C = nn.Parameter(torch.full((n,), _inv_softplus(init_c - self.c_min)))
        self.raw_L = nn.Parameter(torch.full((n,), _inv_softplus(init_l - self.l_min)))

        cg_fraction = (float(init_cg) - self.cg_min) / self.cg_scale
        self.raw_Cj0 = nn.Parameter(torch.full((edges.shape[1],), _logit(cg_fraction)))
        self.raw_Vbias = nn.Parameter(
            torch.full((edges.shape[1],), _inv_softplus(float(init_vbias)))
        )

    @property
    def state_dim(self) -> int:
        """State dimension for ``concat(phi, V)``."""
        return 2 * self.n

    def positive_parameters(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return constrained positive ``C``, ``L``, ``Cj0``, and ``Vbias``."""
        C = self.c_min + F.softplus(self.raw_C)
        L = self.l_min + F.softplus(self.raw_L)
        Cj0 = self.cg_min + self.cg_scale * torch.sigmoid(self.raw_Cj0)
        Vbias = self.vbias_min + F.softplus(self.raw_Vbias)
        return C, L, Cj0, Vbias

    def _autocast_disabled(self, device_type: str):
        if device_type in {"cpu", "cuda", "xpu", "hpu"}:
            return torch.amp.autocast(device_type=device_type, enabled=False)
        return nullcontext()

    def forward(
        self,
        state: Tensor,
        time: Tensor | float | None = None,
        drive: Tensor | None = None,
    ) -> Tensor:
        """Compute ``dstate/dt`` for a batched LC state.

        ``time`` and ``drive`` are accepted for compatibility with other
        dynamics modules but are unused in this passive toy model.
        """
        del time, drive
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError(
                f"state must have shape (batch, {self.state_dim}), got {tuple(state.shape)}."
            )

        input_dtype = state.dtype
        with self._autocast_disabled(state.device.type):
            state_f = state.float()
            phi = state_f[:, : self.n]
            voltage = state_f[:, self.n :]

            C, L, Cj0, Vbias = self.positive_parameters()
            C = C.float()
            L = L.float()
            Cj0 = Cj0.float()
            Vbias = Vbias.float()

            src = self.edges[0]
            dst = self.edges[1]
            edge_voltage = voltage[:, src] - voltage[:, dst]
            v_safe = self.v_clip_scale * torch.tanh(edge_voltage / self.v_clip_scale)
            denom = 1.0 + (Vbias.unsqueeze(0) + v_safe) / self.vj
            denom = denom.clamp_min(1e-4)
            Cg = Cj0.unsqueeze(0) / denom.pow(self.m)

            incidence = self.incidence.float()
            edge_mass = torch.einsum("be,ei,ej->bij", Cg, incidence, incidence)
            node_mass = torch.diag_embed(C).unsqueeze(0)
            eye = torch.eye(self.n, device=state.device, dtype=torch.float32).unsqueeze(0)
            mass = node_mass + edge_mass + self.jitter * eye

            rhs = -phi / L.unsqueeze(0)
            d_voltage = torch.linalg.solve(mass, rhs.unsqueeze(-1)).squeeze(-1)
            derivative = torch.cat([voltage, d_voltage], dim=1)

        return derivative.to(dtype=input_dtype)


def fixed_step_integrate(
    rhs,
    y0: Tensor,
    num_steps: int,
    integration_time: float,
    method: IntegrationMethod = "rk4",
) -> Tensor:
    """Differentiate through a fixed-step Euler or RK4 rollout."""
    if num_steps < 0:
        raise ValueError(f"num_steps must be non-negative, got {num_steps}.")
    if integration_time < 0.0:
        raise ValueError(f"integration_time must be non-negative, got {integration_time}.")
    if method not in ("euler", "rk4"):
        raise ValueError(f"method must be 'euler' or 'rk4', got {method!r}.")
    if num_steps == 0 or integration_time == 0.0:
        return y0

    dt_value = float(integration_time) / int(num_steps)
    dt = y0.new_tensor(dt_value)
    half_dt = y0.new_tensor(0.5 * dt_value)
    y = y0
    t = y0.new_tensor(0.0)
    for _ in range(int(num_steps)):
        if method == "euler":
            y = y + dt * rhs(y, t)
        else:
            k1 = rhs(y, t)
            k2 = rhs(y + half_dt * k1, t + half_dt)
            k3 = rhs(y + half_dt * k2, t + half_dt)
            k4 = rhs(y + dt * k3, t + dt)
            y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t = t + dt
    return y


class _ResizeConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _TinyResizeConvDecoder(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        out_channels: int = 3,
        image_size: int = 32,
        stem_channels: int = 8,
        stem_size: int = 4,
    ) -> None:
        super().__init__()
        if feature_dim != stem_channels * stem_size * stem_size:
            raise ValueError(
                "feature_dim must equal stem_channels * stem_size * stem_size; "
                f"got {feature_dim} and {stem_channels} * {stem_size} * {stem_size}."
            )
        if image_size != 32 or stem_size != 4:
            raise ValueError("The toy decoder currently targets 32x32 images from a 4x4 stem.")

        self.feature_dim = int(feature_dim)
        self.output_dim = int(out_channels * image_size * image_size)
        self.stem_channels = int(stem_channels)
        self.stem_size = int(stem_size)
        self.net = nn.Sequential(
            _ResizeConvBlock(stem_channels, 32),
            _ResizeConvBlock(32, 32),
            _ResizeConvBlock(32, 32),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, a=0.2, nonlinearity="leaky_relu")
                nn.init.zeros_(module.bias)

    def forward(self, features: Tensor) -> Tensor:
        x = features.reshape(features.shape[0], self.stem_channels, self.stem_size, self.stem_size)
        x = self.net(x)
        return x.reshape(features.shape[0], self.output_dim)


class PassiveLCGenerator(nn.Module):
    """Small class-conditional image generator driven by passive LC dynamics."""

    def __init__(
        self,
        *,
        n_oscillators: int = 64,
        num_classes: int = 10,
        topology: Topology = "ring",
        k: int = 4,
        seed: int = 0,
        num_steps: int = 8,
        integration_time: float = 1.0,
        method: IntegrationMethod = "rk4",
        decoder_feature_dim: int = 128,
        image_size: int = 32,
        image_channels: int = 3,
        initial_state_scale: float = 0.1,
        label_only: bool = False,
    ) -> None:
        super().__init__()
        if decoder_feature_dim != 128:
            raise ValueError("decoder_feature_dim must be 128 for the 8x4x4 toy decoder.")
        if num_classes < 1:
            raise ValueError(f"num_classes must be positive, got {num_classes}.")
        if initial_state_scale <= 0.0:
            raise ValueError(
                f"initial_state_scale must be positive, got {initial_state_scale}."
            )

        self.num_classes = int(num_classes)
        self.num_steps = int(num_steps)
        self.integration_time = float(integration_time)
        self.method = method
        self.initial_state_scale = float(initial_state_scale)
        self.image_size = int(image_size)
        self.image_channels = int(image_channels)
        self.label_only = bool(label_only)

        self.dynamics = PassiveLCDynamics(
            n=int(n_oscillators),
            topology=topology,
            k=int(k),
            seed=int(seed),
        )
        self.class_offset = nn.Embedding(self.num_classes, self.dynamics.state_dim)
        nn.init.normal_(self.class_offset.weight, mean=0.0, std=0.02)

        self.readout_norm = nn.LayerNorm(self.dynamics.state_dim)
        self.readout = nn.Linear(self.dynamics.state_dim, decoder_feature_dim)
        self.decoder = _TinyResizeConvDecoder(
            feature_dim=decoder_feature_dim,
            out_channels=self.image_channels,
            image_size=self.image_size,
            stem_channels=8,
            stem_size=4,
        )

    def _sample_initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        return (
            torch.randn(
                batch_size,
                self.dynamics.state_dim,
                device=device,
                dtype=dtype,
                generator=generator,
            )
            * self.initial_state_scale
        )

    def forward(
        self,
        class_id: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Generate flat image tensors in ``[-1, 1]`` for class labels."""
        param = next(self.parameters())
        labels = class_id.to(device=param.device, dtype=torch.long)
        if labels.ndim != 1:
            raise ValueError(f"class_id must be 1D, got shape {tuple(class_id.shape)}.")
        if torch.any(labels < 0) or torch.any(labels >= self.num_classes):
            raise ValueError(f"class_id values must be in [0, {self.num_classes}).")

        if self.label_only:
            final_state = self.class_offset(labels).to(dtype=param.dtype)
        else:
            y0 = self._sample_initial_state(
                int(labels.shape[0]),
                device=param.device,
                dtype=param.dtype,
                generator=generator,
            )
            y0 = y0 + self.class_offset(labels).to(dtype=param.dtype)
            final_state = fixed_step_integrate(
                lambda y, t: self.dynamics(y, t),
                y0,
                num_steps=self.num_steps,
                integration_time=self.integration_time,
                method=self.method,
            )
        features = torch.tanh(self.readout(self.readout_norm(final_state)))
        return self.decoder(features)

    @torch.no_grad()
    def sample(
        self,
        class_id: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Generate flat samples without gradients."""
        was_training = self.training
        self.eval()
        try:
            return self.forward(class_id, generator=generator)
        finally:
            self.train(was_training)

    @torch.no_grad()
    def sample_images(
        self,
        class_id: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Generate image tensors shaped ``(B, C, H, W)`` in ``[0, 1]``."""
        flat = self.sample(class_id, generator=generator)
        images = flat.reshape(-1, self.image_channels, self.image_size, self.image_size)
        return ((images + 1.0) * 0.5).clamp(0.0, 1.0)
