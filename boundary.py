from __future__ import annotations

import itertools

import numpy as np
from constants import S_RATIO, TYPE_AA, TYPE_CG


def find_robust_basis(vectors: np.ndarray, cos_threshold: float = 0.8) -> np.ndarray:
    basis_slots: list[dict] = []
    for vec in vectors[np.argsort(np.linalg.norm(vectors, axis=1))]:
        l_vec = float(np.linalg.norm(vec))
        if l_vec < 1e-3:
            continue
        v_dir = vec / l_vec
        matched = False
        for slot in basis_slots:
            c_vec = slot["v"]
            l_c = float(np.linalg.norm(c_vec))
            cos_angle = float(np.dot(v_dir, c_vec / l_c))
            if abs(cos_angle) <= cos_threshold:
                continue
            matched = True
            sign = np.sign(cos_angle)
            vec_aligned = vec * sign
            ratio = l_vec / l_c
            if ratio >= 1.5:
                vec_aligned = vec_aligned / round(ratio)
            elif ratio <= 0.66:
                slot["v"] = c_vec / round(1 / ratio)
            w = slot["w"]
            slot["v"] = (slot["v"] * w + vec_aligned) / (w + 1)
            slot["w"] += 1
            break
        if not matched:
            if len(basis_slots) < 2:
                basis_slots.append({"v": vec, "w": 1})
            elif len(basis_slots) == 2:
                b1, b2 = basis_slots[0]["v"], basis_slots[1]["v"]
                n = np.cross(b1, b2)
                nn = float(np.linalg.norm(n))
                if nn > 1e-5 and abs(float(np.dot(v_dir, n / nn))) > 0.2:
                    basis_slots.append({"v": vec, "w": 1})
    if len(basis_slots) < 3:
        raise RuntimeError("failed to infer 3 basis vectors")
    m = np.column_stack([slot["v"] for slot in basis_slots])
    if np.linalg.det(m) < 0:
        m[:, [1, 2]] = m[:, [2, 1]]
    return m


def basis_from_cg(t2: np.ndarray, center: np.ndarray) -> np.ndarray | None:
    """Lattice basis from CG positions; works with 1+ CG (no hard minimum of 3)."""
    vecs = t2 - center
    if len(vecs) >= 3:
        return find_robust_basis(vecs)
    if len(vecs) == 2:
        b1, b2 = vecs[0], vecs[1]
        b3 = np.cross(b1, b2)
        if np.linalg.norm(b3) < 1e-5:
            aux = np.array([0.0, 0.0, 1.0])
            if abs(float(np.dot(b1 / (np.linalg.norm(b1) + 1e-12), aux))) > 0.9:
                aux = np.array([1.0, 0.0, 0.0])
            b3 = np.cross(b1, aux)
        scale = 0.5 * (np.linalg.norm(b1) + np.linalg.norm(b2))
        b3 = b3 / (np.linalg.norm(b3) + 1e-12) * scale
        return find_robust_basis(np.vstack([b1, b2, b3]))
    if len(vecs) == 1:
        b1 = vecs[0]
        ln = max(float(np.linalg.norm(b1)), 1e-3)
        b1n = b1 / ln
        aux = np.array([0.0, 0.0, 1.0])
        if abs(float(np.dot(b1n, aux))) > 0.9:
            aux = np.array([1.0, 0.0, 0.0])
        b2 = np.cross(b1n, aux)
        b2 = b2 / (np.linalg.norm(b2) + 1e-12) * ln
        b3 = np.cross(b1n, b2)
        b3 = b3 / (np.linalg.norm(b3) + 1e-12) * ln
        m = np.column_stack([b1, b2, b3])
        if np.linalg.det(m) < 0:
            m[:, [1, 2]] = m[:, [2, 1]]
        return m
    return None


def lattice_offsets_fcc(max_index: int = 2) -> np.ndarray:
    cands = np.array(list(itertools.product(range(-max_index, max_index + 1), repeat=3)), dtype=int)
    mask = np.any(cands != 0, axis=1) & (cands.sum(axis=1) % 2 == 0)
    return cands[mask]


def sphere_mask(pos: np.ndarray, center: np.ndarray, r_cut: float) -> np.ndarray:
    return np.linalg.norm(pos - center, axis=1) < r_cut


def estimate_r0(pos: np.ndarray, n: int = 50) -> float:
    p = pos[: min(n, len(pos))]
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(np.min(d))


def unpack_sphere(
    pos: np.ndarray,
    types: np.ndarray,
    center: np.ndarray,
    r_cut: float,
    r0_fine: float,
    *,
    filter_at: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Situation A: reals in sphere + phantom midpoints (Ag). Returns pos, types, center."""
    filt = filter_at if filter_at is not None else center
    m = sphere_mask(pos, center, r_cut)
    sphere_pos, sphere_types = pos[m], types[m]

    t2 = sphere_pos[sphere_types == TYPE_CG]
    t1 = sphere_pos[sphere_types == TYPE_AA]
    M = basis_from_cg(t2, center)
    if M is None:
        parts = [p for p in (t1, t2) if len(p)]
        out_pos = np.vstack(parts) if parts else np.empty((0, 3))
        out_types = np.array([TYPE_AA] * len(t1) + [TYPE_CG] * len(t2), dtype=int)
        return out_pos, out_types, center

    M_base = M / S_RATIO
    M_inv = np.linalg.inv(M_base)
    t2_lat = center + np.round((t2 - center) @ M_inv.T).astype(int) @ M_base.T

    cart_off = lattice_offsets_fcc(2) @ M_base.T
    cands = np.array([p + off for p in t2_lat for off in cart_off])
    if len(cands):
        cands = np.unique(cands.round(3), axis=0)

    inside = np.linalg.norm(cands - filt, axis=1) < r_cut
    if len(sphere_pos):
        md = np.min(np.linalg.norm(cands[:, None, :] - sphere_pos[None, :, :], axis=-1), axis=1)
    else:
        md = np.full(len(cands), np.inf)
    ph = cands[inside & (md > 0.5 * r0_fine)]

    out_pos = np.vstack([t1, t2] + ([ph] if len(ph) else []))
    out_types = np.array(
        [TYPE_AA] * len(t1) + [TYPE_CG] * len(t2) + [0] * len(ph), dtype=int
    )  # 0 = phantom
    return out_pos, out_types, center


def pack_sphere(
    pos: np.ndarray,
    types: np.ndarray,
    center: np.ndarray,
    r_cut: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Situation B: all CG + surviving fine on coarse nodes (Au)."""
    m = sphere_mask(pos, center, r_cut)
    sphere_pos, sphere_types = pos[m], types[m]
    t2 = sphere_pos[sphere_types == TYPE_CG]
    t1 = sphere_pos[sphere_types == TYPE_AA]

    M = basis_from_cg(t2, center)
    if M is None:
        parts = [p for p in (t2, t1) if len(p)]
        out_pos = np.vstack(parts) if parts else np.empty((0, 3))
        out_types = np.array([TYPE_CG] * len(t2) + [TYPE_AA] * len(t1), dtype=int)
        return out_pos, out_types, center

    M_base = M / S_RATIO
    M_inv = np.linalg.inv(M_base)
    coords = (t1 - center) @ M_inv.T
    ci = np.round(coords).astype(int)
    keep = (
        (ci[:, 0] % 2 == 0)
        & (ci[:, 1] % 2 == 0)
        & (ci[:, 2] % 2 == 0)
        & (ci.sum(axis=1) % 2 == 0)
        & (np.linalg.norm(coords - ci, axis=1) < 0.35)
    )
    surv = t1[keep]
    out_pos = np.vstack([t2, surv]) if len(surv) else t2
    out_types = np.array([TYPE_CG] * len(t2) + [TYPE_AA] * len(surv), dtype=int)
    return out_pos, out_types, center
