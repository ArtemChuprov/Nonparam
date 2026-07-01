#!/usr/bin/env python3
"""
Minimal bilayer AA|CG demo: ASE geometry (boundary_test), jax-md homogeneous energy, cross in energy_fn.

Run:  python bilayer_cross.py
Debug: F5 → "bilayer_cross" in VS Code
"""

from __future__ import annotations

import logging

import jax.numpy as jnp
import numpy as np
from ase import Atoms
from ase.io import write
from jax_md import space

from boundary import estimate_r0, pack_sphere, sphere_mask, unpack_sphere
from constants import TYPE_AA, TYPE_CG
from energy import build_cross_topology, build_hybrid_energy
from simulation import one_jax_step, save_neighbors, write_xyz
from structure import build_ase_bilayer, orthorhombic_box, symbols_to_types

log = logging.getLogger(__name__)


def log_sphere(
    label: str,
    pos: np.ndarray,
    types: np.ndarray,
    out_path: str,
    fine_sym: str,
    coarse_sym: str,
    *,
    phantom_sym: str = "Ag",
) -> None:
    n_aa = int(np.sum(types == TYPE_AA))
    n_cg = int(np.sum(types == TYPE_CG))
    n_ph = int(np.sum(types == 0))
    log.info("%s: n=%d  AA=%d  CG=%d  phantom=%d  -> %s", label, len(pos), n_aa, n_cg, n_ph, out_path)
    sym = [
        fine_sym if t == TYPE_AA else coarse_sym if t == TYPE_CG else phantom_sym
        for t in types
    ]
    write(out_path, Atoms(symbols=sym, positions=pos))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = "out"

    log.info("=== build ASE bilayer (boundary_test layout) ===")
    atoms, fine_sym, coarse_sym = build_ase_bilayer(out_dir=out)
    pos = atoms.get_positions()
    types = symbols_to_types(atoms.get_chemical_symbols(), fine_sym, coarse_sym)

    r0_fine = estimate_r0(pos[types == TYPE_AA])
    r0_coarse = estimate_r0(pos[types == TYPE_CG])
    r_cut = (1.2 if fine_sym == "Al" else 2.0) * r0_coarse
    log.info("r0_fine=%.3f  r0_coarse=%.3f  R_CUT=%.3f", r0_fine, r0_coarse, r_cut)

    cx = 0.5 * (pos[:, 0].max() + pos[:, 0].min())
    cy = 0.5 * (pos[:, 1].max() + pos[:, 1].min())
    fine_pos = pos[types == TYPE_AA]
    coarse_pos = pos[types == TYPE_CG]
    target_a = np.array([cx, cy, fine_pos[:, 2].max()])
    center_a = fine_pos[np.argmin(np.linalg.norm(fine_pos - target_a, axis=1))]
    target_b = np.array([cx, cy, coarse_pos[:, 2].min()])
    center_b = coarse_pos[np.argmin(np.linalg.norm(coarse_pos - target_b, axis=1))]

    m_a = sphere_mask(pos, center_a, r_cut)
    log.info(
        "sphere A BEFORE: n=%d  AA=%d  CG=%d",
        m_a.sum(), int(np.sum(types[m_a] == TYPE_AA)), int(np.sum(types[m_a] == TYPE_CG)),
    )
    syms = np.array(atoms.get_chemical_symbols())
    write(f"{out}/2_sphere_A_before.xyz", Atoms(symbols=syms[m_a], positions=pos[m_a]))

    pos_a, typ_a, _ = unpack_sphere(pos, types, center_a, r_cut, r0_fine, filter_at=target_a)
    log_sphere("sphere A AFTER (unpack)", pos_a, typ_a, f"{out}/3_sphere_A_after.xyz", fine_sym, coarse_sym)

    m_b = sphere_mask(pos, center_b, r_cut)
    log.info(
        "sphere B BEFORE: n=%d  AA=%d  CG=%d",
        m_b.sum(), int(np.sum(types[m_b] == TYPE_AA)), int(np.sum(types[m_b] == TYPE_CG)),
    )
    write(f"{out}/4_sphere_B_before.xyz", Atoms(symbols=syms[m_b], positions=pos[m_b]))

    pos_b, typ_b, _ = pack_sphere(pos, types, center_b, r_cut)
    n_cg_b = int(np.sum(typ_b == TYPE_CG))
    sym_b = [coarse_sym] * n_cg_b + ["Au"] * (len(pos_b) - n_cg_b)
    log.info(
        "sphere B AFTER (pack): n=%d  CG=%d  surviving_fine=%d  -> %s",
        len(pos_b), n_cg_b, len(pos_b) - n_cg_b, f"{out}/5_sphere_B_after.xyz",
    )
    write(f"{out}/5_sphere_B_after.xyz", Atoms(symbols=sym_b, positions=pos_b))

    log.info("=== jax-md energies ===")
    box_diag = orthorhombic_box(pos)
    R = jnp.array(pos, dtype=jnp.float64)
    types_j = jnp.array(types, dtype=jnp.int32)
    displacement, _ = space.periodic(box_diag)
    nbr_fn, e_homog_fn = build_hybrid_energy(displacement, box_diag, types_j, None)
    E_homog = float(e_homog_fn(R, nbr_fn.allocate(R)))
    log.info("E_homogeneous (AA LJ + CG scale_wrapper): %.6f", E_homog)

    topo = build_cross_topology(pos, types, r_cut, r0_fine, target_a)
    log.info(
        "cross topology: unpack_centers=%d  pack_centers=%d  ph_total=%d",
        len(topo.unpack_centers),
        len(topo.pack_centers),
        sum(len(o) for o in topo.unpack_offs),
    )

    _, e_hybrid_fn = build_hybrid_energy(displacement, box_diag, types_j, topo)
    neighbor = nbr_fn.allocate(R)
    E_total = float(e_hybrid_fn(R, neighbor))
    log.info("E_cross (approx): %.6f", E_total - E_homog)
    log.info("E_total (homog + cross): %.6f", E_total)

    log.info("=== one minimal JAX simulation step ===")
    # =====================================================================
    # DEBUG HERE: one JAX step uses the full hybrid energy above.
    # Inspect neighbor files before/after this block if cross behavior jumps.
    # =====================================================================
    save_neighbors(f"{out}/neighbors_before.txt", neighbor, types)
    R1, _, neighbor1, E_before, E_after = one_jax_step(
        R, jnp.zeros_like(R), neighbor, nbr_fn, e_hybrid_fn, dt=1e-4
    )
    save_neighbors(f"{out}/neighbors_after.txt", neighbor1, types)
    write_xyz(f"{out}/6_after_jax_step.xyz", R1, atoms.get_chemical_symbols())
    log.info("E_before_step: %.6f", float(E_before))
    log.info("E_after_step: %.6f", float(E_after))
    log.info("max_displacement_after_step: %.6e", float(jnp.max(jnp.linalg.norm(R1 - R, axis=1))))
    log.info("saved: out/neighbors_before.txt, out/neighbors_after.txt, out/6_after_jax_step.xyz")


if __name__ == "__main__":
    main()
