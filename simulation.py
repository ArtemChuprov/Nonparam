from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from ase import Atoms
from ase.io import write


def neighbor_rows(neighbor_idx, n_atoms: int) -> list[list[int]]:
    idx = np.asarray(neighbor_idx)
    if idx.shape[0] != n_atoms and idx.shape[-1] == n_atoms:
        idx = idx.T
    return [[int(j) for j in row if 0 <= int(j) < n_atoms] for row in idx]


def save_neighbors(path: str, neighbor, types: np.ndarray) -> None:
    rows = neighbor_rows(neighbor.idx, len(types))
    with open(path, "w", encoding="utf-8") as f:
        for i, js in enumerate(rows):
            f.write(f"{i} type={int(types[i])} n={len(js)}: {' '.join(map(str, js))}\n")


def one_jax_step(R, V, neighbor, nbr_fn, energy_fn, *, dt: float = 1e-5):
    """One minimal velocity-Verlet step with unit masses."""

    force_fn = jax.grad(lambda X, nbr: energy_fn(X, nbr))
    E0 = energy_fn(R, neighbor)
    F0 = -force_fn(R, neighbor)
    R1 = R + dt * V + 0.5 * dt * dt * F0
    neighbor1 = nbr_fn.allocate(R1)
    F1 = -force_fn(R1, neighbor1)
    V1 = V + 0.5 * dt * (F0 + F1)
    E1 = energy_fn(R1, neighbor1)
    return R1, V1, neighbor1, E0, E1


def write_xyz(path: str, R, symbols: list[str]) -> None:
    write(path, Atoms(symbols=symbols, positions=np.asarray(jnp.asarray(R))))
