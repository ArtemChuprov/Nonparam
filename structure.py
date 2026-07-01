from __future__ import annotations

import os

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import write

from constants import TYPE_AA, TYPE_CG


def build_ase_bilayer(
    *,
    lattice_type: str = "fcc",
    seed: int = 0,
    out_dir: str = "out",
) -> tuple[Atoms, str, str]:
    rng = np.random.default_rng(seed)
    if lattice_type == "fcc":
        a = 4.05
        fine = bulk("Al", cubic=True).repeat((8, 8, 4))
        coarse = bulk("Al", cubic=True).repeat((4, 4, 2))
        fine_sym, coarse_sym = "Al", "Cu"
    else:
        a = 3.57
        fine = bulk("C", "diamond", cubic=True).repeat((8, 8, 4))
        coarse = bulk("C", "diamond", cubic=True).repeat((4, 4, 2))
        fine_sym, coarse_sym = "C", "Si"

    coarse.set_positions(coarse.get_positions() * 2.0)
    coarse.set_chemical_symbols([coarse_sym] * len(coarse))
    coarse.positions[:, 2] += np.max(fine.positions[:, 2]) + 0.5 * a

    system = Atoms(
        symbols=fine.get_chemical_symbols() + coarse.get_chemical_symbols(),
        positions=np.vstack([fine.positions, coarse.positions]),
    )
    theta = np.pi / 6
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]).T
    system.positions = system.positions @ R
    system.positions[:, 0] *= 1.05
    system.positions += rng.normal(0, 0.04, system.positions.shape)

    os.makedirs(out_dir, exist_ok=True)
    write(f"{out_dir}/1_full.xyz", system)
    return system, fine_sym, coarse_sym


def symbols_to_types(symbols: list[str], fine_sym: str, coarse_sym: str) -> np.ndarray:
    m = {fine_sym: TYPE_AA, coarse_sym: TYPE_CG}
    return np.array([m[s] for s in symbols], dtype=np.int32)


def orthorhombic_box(pos: np.ndarray, pad: float = 3.0) -> np.ndarray:
    lo = pos.min(axis=0) - pad
    hi = pos.max(axis=0) + pad
    return hi - lo
