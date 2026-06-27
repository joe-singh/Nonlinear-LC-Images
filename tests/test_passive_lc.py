from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from un0.passive_lc import (
    PassiveLCDynamics,
    PassiveLCGenerator,
    fixed_step_integrate,
    make_lc_edges,
)


def _load_toy_fid_script():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "eval_passive_lc_toy_fid.py"
    spec = importlib.util.spec_from_file_location("eval_passive_lc_toy_fid", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_toy_train_script():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "train_passive_lc_toy_cifar10.py"
    )
    spec = importlib.util.spec_from_file_location("train_passive_lc_toy_cifar10", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assert_valid_edges(edges: torch.Tensor, n: int) -> None:
    assert edges.dtype == torch.long
    assert edges.ndim == 2
    assert edges.shape[0] == 2
    assert edges.shape[1] > 0
    assert int(edges.min()) >= 0
    assert int(edges.max()) < n
    assert torch.all(edges[0] != edges[1])

    undirected = {tuple(sorted(pair)) for pair in edges.t().tolist()}
    assert len(undirected) == edges.shape[1]


def test_make_lc_edges_valid_topologies() -> None:
    n = 8
    ring = make_lc_edges(n, topology="ring")
    chain = make_lc_edges(n, topology="chain")
    sparse_a = make_lc_edges(n, topology="random_sparse", k=4, seed=123)
    sparse_b = make_lc_edges(n, topology="random_sparse", k=4, seed=123)
    all_to_all = make_lc_edges(n, topology="all_to_all")

    _assert_valid_edges(ring, n)
    _assert_valid_edges(chain, n)
    _assert_valid_edges(sparse_a, n)
    _assert_valid_edges(all_to_all, n)
    assert ring.shape[1] == n
    assert chain.shape[1] == n - 1
    assert sparse_a.shape[1] == round(n * 4 / 2)
    assert all_to_all.shape[1] == n * (n - 1) // 2
    assert torch.equal(sparse_a, sparse_b)


def test_passive_lc_dynamics_forward_shape_and_finite() -> None:
    torch.manual_seed(0)
    dynamics = PassiveLCDynamics(n=8, topology="ring")
    state = 0.1 * torch.randn(4, dynamics.state_dim)

    derivative = dynamics(state)

    assert derivative.shape == (4, dynamics.state_dim)
    assert torch.isfinite(derivative).all()


def test_fixed_step_rollout_stays_finite() -> None:
    torch.manual_seed(1)
    dynamics = PassiveLCDynamics(n=8, topology="random_sparse", k=3, seed=7)
    state = 0.1 * torch.randn(4, dynamics.state_dim)

    final_state = fixed_step_integrate(
        lambda y, t: dynamics(y, t),
        state,
        num_steps=2,
        integration_time=0.5,
        method="rk4",
    )

    assert final_state.shape == state.shape
    assert torch.isfinite(final_state).all()


def test_passive_lc_generator_returns_flat_cifar_images() -> None:
    torch.manual_seed(2)
    model = PassiveLCGenerator(n_oscillators=8, num_steps=2, integration_time=0.5)
    labels = torch.tensor([0, 1, 2, 3])

    samples = model(labels)

    assert samples.shape == (4, 3 * 32 * 32)
    assert torch.isfinite(samples).all()


def test_passive_lc_generator_accepts_narrow_decoder() -> None:
    torch.manual_seed(22)
    model = PassiveLCGenerator(
        n_oscillators=8,
        num_steps=0,
        decoder_width=8,
    )
    labels = torch.tensor([0, 1, 2, 3])

    samples = model(labels)

    assert model.decoder.hidden_channels == 8
    assert samples.shape == (4, 3 * 32 * 32)
    assert torch.isfinite(samples).all()


def test_passive_lc_generator_accepts_linear_decoder() -> None:
    torch.manual_seed(23)
    model = PassiveLCGenerator(
        n_oscillators=8,
        num_steps=0,
        decoder_type="linear",
    )
    labels = torch.tensor([0, 1, 2, 3])

    samples = model(labels)

    assert model.decoder_type == "linear"
    assert model.readout.out_features == 3 * 32 * 32
    assert samples.shape == (4, 3 * 32 * 32)
    assert torch.isfinite(samples).all()


def test_passive_lc_generator_backward_has_finite_gradients() -> None:
    torch.manual_seed(3)
    model = PassiveLCGenerator(n_oscillators=8, num_steps=2, integration_time=0.5)
    labels = torch.tensor([0, 1, 2, 3])

    samples = model(labels)
    loss = samples.square().mean()
    loss.backward()

    finite_grad_sums = [
        float(param.grad.detach().abs().sum())
        for param in model.parameters()
        if param.requires_grad and param.grad is not None and torch.isfinite(param.grad).all()
    ]
    assert finite_grad_sums
    assert any(value > 0.0 for value in finite_grad_sums)


def test_label_only_generator_skips_dynamics() -> None:
    torch.manual_seed(4)
    model = PassiveLCGenerator(n_oscillators=8, num_steps=2, label_only=True)
    labels = torch.tensor([0, 1, 0, 1])

    samples_a = model(labels)
    samples_b = model(labels)
    loss = samples_a.square().mean()
    loss.backward()

    assert samples_a.shape == (4, 3 * 32 * 32)
    torch.testing.assert_close(samples_a, samples_b)
    assert torch.isfinite(samples_a).all()
    assert model.class_offset.weight.grad is not None
    assert all(parameter.grad is None for parameter in model.dynamics.parameters())


def test_fid_eval_rebuilds_model_from_training_args() -> None:
    module = _load_toy_fid_script()
    ckpt_args = {
        "n_oscillators": 8,
        "topology": "chain",
        "k": 2,
        "seed": 9,
        "num_steps": 0,
        "integration_time": 0.75,
        "method": "euler",
        "label_only": True,
        "decoder_width": 8,
        "decoder_type": "linear",
    }

    model = module.build_model_from_checkpoint_args(
        ckpt_args,
        device=torch.device("cpu"),
    )

    assert model.dynamics.n == 8
    assert model.dynamics.topology == "chain"
    assert model.num_steps == 0
    assert model.integration_time == 0.75
    assert model.method == "euler"
    assert model.label_only
    assert model.decoder_type == "linear"
    assert model.readout.out_features == 3 * 32 * 32


def test_training_diagnostics_track_lc_delta_and_grad_norms() -> None:
    module = _load_toy_train_script()
    torch.manual_seed(5)
    model = PassiveLCGenerator(n_oscillators=8, num_steps=2, integration_time=0.5)
    labels = torch.tensor([0, 1, 2, 3])
    reference = module.lc_parameter_snapshot(model)

    initial_delta = module.lc_parameter_delta_summary(model, reference)
    assert initial_delta["C"]["mean_abs"] == 0.0
    assert initial_delta["L"]["mean_abs"] == 0.0
    assert initial_delta["Cj0"]["mean_abs"] == 0.0
    assert initial_delta["Vbias"]["mean_abs"] == 0.0

    loss = model(labels).square().mean()
    loss.backward()
    grad_norms = module.module_gradient_norms(model)

    assert grad_norms["dynamics"] > 0.0
    assert grad_norms["decoder"] > 0.0
    assert all(torch.isfinite(torch.tensor(value)) for value in grad_norms.values())

    with torch.no_grad():
        model.dynamics.raw_C.add_(0.01)
    moved_delta = module.lc_parameter_delta_summary(model, reference)
    assert moved_delta["C"]["mean_abs"] > 0.0


def test_training_optimizer_can_boost_dynamics_lr() -> None:
    module = _load_toy_train_script()
    model = PassiveLCGenerator(n_oscillators=8, num_steps=2, integration_time=0.5)

    optimizer, trainable, optimizer_lrs = module.build_optimizer(
        model,
        lr=1e-3,
        dynamics_lr_multiplier=10.0,
    )

    assert trainable
    assert optimizer_lrs["dynamics"] == 1e-2
    assert optimizer_lrs["non_dynamics"] == 1e-3
    assert len(optimizer.param_groups) == 2
    assert sorted(group["name"] for group in optimizer.param_groups) == [
        "dynamics",
        "non_dynamics",
    ]


def test_training_optimizer_omits_frozen_dynamics_group() -> None:
    module = _load_toy_train_script()
    model = PassiveLCGenerator(n_oscillators=8, num_steps=2, integration_time=0.5)
    for parameter in model.dynamics.parameters():
        parameter.requires_grad_(False)

    optimizer, trainable, optimizer_lrs = module.build_optimizer(
        model,
        lr=1e-3,
        dynamics_lr_multiplier=10.0,
    )

    assert trainable
    assert optimizer_lrs["dynamics"] == 0.0
    assert optimizer_lrs["non_dynamics"] == 1e-3
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["name"] == "non_dynamics"
