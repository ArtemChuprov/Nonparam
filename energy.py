from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax_md import partition, smap, space

from boundary import find_robust_basis, sphere_mask, unpack_sphere
from constants import EPS, NBR_CUT, R_CUT, SCALE_A_CG, SCALE_B_CG, SIG, S_RATIO, TYPE_AA, TYPE_CG

jax.config.update("jax_enable_x64", True)


def lj_radial(d, sigma=SIG, epsilon=EPS):
    d = jnp.maximum(d, 1e-10)
    inv = sigma / d
    inv6 = inv**6
    return 4.0 * epsilon * (inv6**2 - inv6)


def radial_pair(radial_energy):
    """Adapt U(d) to jax-md pair form U(dr)."""

    def pair_energy(dr, **kwargs):
        return radial_energy(space.distance(dr), **kwargs)

    return pair_energy


def with_cutoff(pair_energy):
    """Zero a pair potential by physical distance of the displacement passed to it."""

    def cut_pair_energy(dr, r_cutoff, **kwargs):
        d = space.distance(dr)
        return jnp.where(d < r_cutoff, pair_energy(dr, **kwargs), 0.0)

    return cut_pair_energy


def scale_wrapper(pair_energy):
    """Generic scale wrapper: U_scaled(dr) = A * U_base(B * dr).

    `pair_energy` can be LJ, EAM-like pair part, or any function with jax-md
    pair signature `pair_energy(dr, **params)`. Cutoffs inside `pair_energy`
    are evaluated on the scaled displacement `B * dr`.
    """

    def scaled_pair_energy(dr, A=1.0, B=1.0, **kwargs):
        B_dr = jnp.asarray(B)
        while B_dr.ndim < dr.ndim:
            B_dr = B_dr[..., None]
        return jnp.where(jnp.abs(A) > 1e-12, A * pair_energy(B_dr * dr, **kwargs), 0.0)

    return scaled_pair_energy


lj_pair = with_cutoff(radial_pair(lj_radial))
scaled_lj_pair = scale_wrapper(lj_pair)


def pair_scaled_lj(dr, sigma, epsilon, r_cutoff, A, B):
    # Backward-compatible name; actual wrapper is generic above.
    return scaled_lj_pair(dr, sigma=sigma, epsilon=epsilon, r_cutoff=r_cutoff, A=A, B=B)


@dataclass
class CrossTopology:
    """Frozen unpack/pack topology at R0; phantom offsets relative to center."""

    unpack_centers: tuple[int, ...]
    unpack_offs: tuple[tuple[np.ndarray, ...], ...]  # per center list of (3,) offsets
    pack_centers: tuple[int, ...]
    pack_fine: tuple[tuple[int, ...], ...]  # per CG center, global fine indices


def pack_fine_indices(pos: np.ndarray, types: np.ndarray, center_i: int, r_cut: float) -> tuple[int, ...]:
    m = sphere_mask(pos, pos[center_i], r_cut)
    sp, st = pos[m], types[m]
    t2, t1 = sp[st == TYPE_CG], sp[st == TYPE_AA]
    g_idx = np.where(m)[0]
    t1_g = g_idx[st == TYPE_AA]
    if len(t2) < 3:
        return ()
    M = find_robust_basis(t2 - pos[center_i])
    Mb = M / S_RATIO
    Mi = np.linalg.inv(Mb)
    coords = (t1 - pos[center_i]) @ Mi.T
    ci = np.round(coords).astype(int)
    keep = (
        (ci[:, 0] % 2 == 0)
        & (ci[:, 1] % 2 == 0)
        & (ci[:, 2] % 2 == 0)
        & (ci.sum(axis=1) % 2 == 0)
        & (np.linalg.norm(coords - ci, axis=1) < 0.35)
    )
    return tuple(int(t1_g[k]) for k in np.where(keep)[0])


def build_cross_topology(
    pos: np.ndarray,
    types: np.ndarray,
    r_cut: float,
    r0_fine: float,
    filter_at: np.ndarray,
) -> CrossTopology:
    """All interface AA/CG with cross neighbors; topology frozen at R0."""
    unpack_centers, unpack_offs_list = [], []
    for i in range(len(pos)):
        if types[i] != TYPE_AA:
            continue
        if not np.any((types == TYPE_CG) & sphere_mask(pos, pos[i], r_cut)):
            continue
        pos_u, typ_u, _ = unpack_sphere(pos, types, pos[i], r_cut, r0_fine, filter_at=filter_at)
        ph = pos_u[typ_u == 0]
        if len(ph):
            unpack_centers.append(i)
            unpack_offs_list.append(tuple(ph - pos[i]))

    pack_centers, pack_fine_list = [], []
    for i in range(len(pos)):
        if types[i] != TYPE_CG:
            continue
        if not np.any((types == TYPE_AA) & sphere_mask(pos, pos[i], r_cut)):
            continue
        fine_idx = pack_fine_indices(pos, types, i, r_cut)
        if fine_idx:
            pack_centers.append(i)
            pack_fine_list.append(fine_idx)

    return CrossTopology(
        tuple(unpack_centers),
        tuple(unpack_offs_list),
        tuple(pack_centers),
        tuple(pack_fine_list),
    )


def cross_energy(R: jnp.ndarray, displacement, topo: CrossTopology) -> jnp.ndarray:
    """Path 1: LJ(center, phantom) + scaled LJ(CG, fine); phantoms co-move with center."""
    e = jnp.array(0.0, dtype=jnp.float64)
    for i, offs in zip(topo.unpack_centers, topo.unpack_offs):
        rc = R[i]
        for off in offs:
            ph = rc + jnp.asarray(off)
            dr = displacement(rc, ph)
            d = space.distance(dr)
            e = e + jnp.where((d > 1e-8) & (d < R_CUT), lj_radial(d), 0.0)
    for i, fine_idx in zip(topo.pack_centers, topo.pack_fine):
        rc = R[i]
        for j in fine_idx:
            dr = displacement(rc, R[j])
            e = e + jnp.where(
                space.distance(dr) > 1e-8,
                scaled_lj_pair(dr, sigma=SIG, epsilon=EPS, r_cutoff=R_CUT, A=SCALE_A_CG, B=SCALE_B_CG),
                0.0,
            )
    return e


def build_hybrid_energy(displacement, box, types: jnp.ndarray, topo: CrossTopology | None):
    species = jnp.asarray(types, dtype=jnp.int32)
    A = jnp.zeros((3, 3)).at[1, 1].set(1.0).at[2, 2].set(SCALE_A_CG)
    B = jnp.zeros((3, 3)).at[1, 1].set(1.0).at[2, 2].set(SCALE_B_CG)
    nbr_fn = partition.neighbor_list(
        displacement, box, NBR_CUT, dr_threshold=0.25, fractional_coordinates=False
    )
    e_homog = smap.pair_neighbor_list(
        pair_scaled_lj, displacement, species=species,
        sigma=SIG, epsilon=EPS, r_cutoff=R_CUT, A=A, B=B,
    )

    def energy_fn(R, neighbor):
        e = e_homog(R, neighbor)
        # =====================================================================
        # DEBUG HERE: CROSS STRUCTURE IS INSERTED INSIDE energy_fn(R, neighbor).
        # This is the "Path 1" point: boundary topology was built at R0, but
        # phantom/fine cross energy is evaluated from the current JAX positions R.
        # =====================================================================
        if topo is not None:
            e = e + cross_energy(R, displacement, topo)
        return e

    return nbr_fn, energy_fn
