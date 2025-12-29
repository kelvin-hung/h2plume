# ve_core.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


DEFAULT_PARAMS = {
    # diffusion strength and heterogeneity scaling
    "D0": 0.15,
    "alpha_k": 1.0,       # diffusion ~ k_norm**alpha_k
    "dt_scale": 1.0,      # additional multiplier on effective dt

    # source / sink spread
    "src_sigma": 1.2,     # gaussian-ish spread (grid cells)

    # Land trapping parameters
    "Swr": 0.2,
    "Sgr_max": 0.35,
    "C_L": 2.0,           # Land coefficient (bigger -> stronger residual)

    # stabilizers
    "eps": 1e-8,
    "clip_sg": 1.0,
}


@dataclass
class ForwardResult:
    t: np.ndarray
    q: np.ndarray
    sg_list: list[np.ndarray]
    p_list: list[np.ndarray] | None
    area: np.ndarray
    r_eq: np.ndarray


def prepare_phi_k(phi: np.ndarray, k: np.ndarray):
    phi = np.array(phi, dtype=np.float32)
    k = np.array(k, dtype=np.float32)
    if phi.ndim != 2 or k.ndim != 2:
        raise ValueError("phi and k must be 2D arrays (nx, ny).")
    if phi.shape != k.shape:
        raise ValueError(f"phi shape {phi.shape} != k shape {k.shape}")

    mask = np.isfinite(phi) & np.isfinite(k) & (phi > 0)
    if mask.sum() < 10:
        raise ValueError("Too few valid cells in phi/k (mask too small).")

    # Normalize permeability robustly using log-scale
    k_valid = k[mask]
    k_log = np.log10(np.maximum(k_valid, 1e-12))
    lo, hi = np.percentile(k_log, [2, 98])
    k_norm = np.zeros_like(k, dtype=np.float32)
    k_norm[mask] = np.clip((np.log10(np.maximum(k[mask], 1e-12)) - lo) / (hi - lo + 1e-8), 0.0, 1.0)

    # Porosity normalized to typical range (do not over-normalize)
    phi_norm = np.zeros_like(phi, dtype=np.float32)
    pv = phi[mask]
    plo, phi_hi = np.percentile(pv, [2, 98])
    phi_norm[mask] = np.clip((phi[mask] - plo) / (phi_hi - plo + 1e-8), 0.0, 1.0)

    return phi_norm, k_norm, mask


def choose_well_ij(k_norm: np.ndarray, mask: np.ndarray, mode: str, ij=None):
    nx, ny = k_norm.shape
    if mode == "manual":
        if ij is None:
            raise ValueError("manual mode requires ij")
        i, j = int(ij[0]), int(ij[1])
        i = int(np.clip(i, 0, nx - 1))
        j = int(np.clip(j, 0, ny - 1))
        if not mask[i, j]:
            # find nearest active cell
            coords = np.argwhere(mask)
            d2 = (coords[:, 0] - i) ** 2 + (coords[:, 1] - j) ** 2
            m = int(np.argmin(d2))
            i, j = int(coords[m, 0]), int(coords[m, 1])
        return i, j

    if mode == "center":
        i, j = nx // 2, ny // 2
        if mask[i, j]:
            return i, j
        coords = np.argwhere(mask)
        d2 = (coords[:, 0] - i) ** 2 + (coords[:, 1] - j) ** 2
        m = int(np.argmin(d2))
        return int(coords[m, 0]), int(coords[m, 1])

    if mode == "max_k":
        kk = np.where(mask, k_norm, -1.0)
        idx = int(np.argmax(kk))
        i, j = np.unravel_index(idx, kk.shape)
        return int(i), int(j)

    raise ValueError("mode must be one of: max_k, center, manual")


def _diffuse_step(sg: np.ndarray, D: np.ndarray, mask: np.ndarray):
    # 5-point Laplacian with zero-flux boundaries
    s = np.where(mask, sg, 0.0).astype(np.float32)

    s_up = np.roll(s, -1, axis=0); s_up[-1, :] = s[-1, :]
    s_dn = np.roll(s,  1, axis=0); s_dn[0,  :] = s[0,  :]
    s_rt = np.roll(s, -1, axis=1); s_rt[:, -1] = s[:, -1]
    s_lt = np.roll(s,  1, axis=1); s_lt[:,  0] = s[:, 0]

    lap = (s_up + s_dn + s_rt + s_lt - 4.0 * s).astype(np.float32)
    return (D * lap).astype(np.float32)


def _make_source(nx, ny, wi, wj, sigma: float):
    # simple gaussian source kernel
    ii = np.arange(nx, dtype=np.float32)[:, None]
    jj = np.arange(ny, dtype=np.float32)[None, :]
    r2 = (ii - wi) ** 2 + (jj - wj) ** 2
    ker = np.exp(-0.5 * r2 / (sigma ** 2 + 1e-8)).astype(np.float32)
    ker /= np.sum(ker) + 1e-8
    return ker


def _land_residual(Sg: np.ndarray, Sgr_max: float, C_L: float):
    """
    Land model (one common form):
      Sgr = Sgr_max * Sg / (Sg + C_L*(1 - Sg))
    """
    eps = 1e-8
    denom = (Sg + C_L * (1.0 - Sg) + eps).astype(np.float32)
    return (Sgr_max * Sg / denom).astype(np.float32)


def run_forward(
    phi: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    q: np.ndarray,
    params: dict,
    well_mode: str = "max_k",
    well_ij=None,
    return_pressure: bool = True,
    thr_area: float = 0.05,
) -> ForwardResult:
    p = dict(DEFAULT_PARAMS)
    p.update(params or {})

    phi_n, k_n, mask = prepare_phi_k(phi, k)
    nx, ny = phi.shape
    wi, wj = choose_well_ij(k_n, mask, well_mode, ij=well_ij)

    # effective diffusion field
    D = (p["D0"] * (k_n ** float(p["alpha_k"])) + 1e-6).astype(np.float32)
    D = np.where(mask, D, 0.0).astype(np.float32)

    # source kernel
    src = _make_source(nx, ny, wi, wj, float(p["src_sigma"]))
    src = np.where(mask, src, 0.0).astype(np.float32)
    src /= np.sum(src) + 1e-8

    # initialize fields
    sg = np.zeros((nx, ny), dtype=np.float32)
    sg_trap = np.zeros((nx, ny), dtype=np.float32)

    sg_list: list[np.ndarray] = []
    p_list: list[np.ndarray] | None = [] if return_pressure else None

    t = np.array(t, dtype=np.float32).reshape(-1)
    q = np.array(q, dtype=np.float32).reshape(-1)
    if len(t) != len(q):
        raise ValueError("t and q must have same length")

    # Estimate dt in "steps" (if t is days it still works)
    dt = np.diff(t, prepend=t[0]).astype(np.float32)
    dt[0] = dt[1] if len(dt) > 1 else 1.0
    dt = np.maximum(dt, 1e-6) * float(p["dt_scale"])

    for n in range(len(t)):
        # diffusion
        sg = sg + dt[n] * _diffuse_step(sg, D, mask)

        # source/sink: inject adds, produce removes
        # scale by phi to mimic pore-volume response
        inj = (q[n] * src / (phi_n + float(p["eps"]))).astype(np.float32)
        sg = sg + dt[n] * inj

        # apply Land trapping only when flow is not injecting (shut/production)
        if q[n] <= 0.0:
            sigr = _land_residual(np.clip(sg, 0.0, 1.0), float(p["Sgr_max"]), float(p["C_L"]))
            sg_trap = np.maximum(sg_trap, sigr)

        # enforce trapped residual
        sg = np.maximum(sg, sg_trap)

        # clamp and mask
        sg = np.clip(sg, 0.0, float(p["clip_sg"])).astype(np.float32)
        sg = np.where(mask, sg, np.nan).astype(np.float32)

        sg_list.append(sg.copy())

        if return_pressure and p_list is not None:
            # simple pressure surrogate: proportional to cumulative source convolved by src
            # (placeholder; swap with paper pressure later)
            pfield = (sg.copy())
            p_list.append(pfield)

    # area and radius metrics
    area = np.zeros(len(t), dtype=np.float32)
    r_eq = np.zeros(len(t), dtype=np.float32)
    for i, s in enumerate(sg_list):
        a = float(np.sum(np.isfinite(s) & (s > float(thr_area))))
        area[i] = a
        r_eq[i] = np.sqrt(a / np.pi) if a > 0 else 0.0

    return ForwardResult(
        t=t.astype(np.float32),
        q=q.astype(np.float32),
        sg_list=sg_list,
        p_list=p_list,
        area=area,
        r_eq=r_eq,
    )
