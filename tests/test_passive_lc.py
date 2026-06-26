from __future__ import annotations

import torch

from un0.passive_lc import (
    PassiveLCDynamics,
    PassiveLCGenerator,
    fixed_step_integrate,
    make_lc_edges,
)


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
