# ve_core.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

DEFAULT_PARAMS: Dict[str, float] = {
    "D0": 0.20,
    "anisD": 1.0,
    "mob_exp": 1.0,
    "alpha_p": 1.0,
    "ap_diff": 0.7,
    "nu": 0.35,
    "src_sigma": 2.0,
    "prod_sigma": 3.0,
    "prod_frac": 0.85,
    "Swr": 0.15,
    "Sgr_max": 0.35,
    "C_L": 0.25,
    "dt": 0.35,
    "blur_steps": 1,
    "clip_eps": 1e-6,
}

@dataclass
class ForwardResult:
    t: np.ndarray
    q: np.ndarray
    sg_list: List[np.ndarray]
    p_list: Optional[List[np.ndarray]]
    area: np.ndarray
    r_eq: np.ndarray
    well_ij: Tuple[int, int]

def prepare_phi_k(phi: np.ndarray, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi = np.asarray(phi, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    if phi.ndim != 2 or k.ndim != 2:
        raise ValueError("phi and k must be 2D arrays.")
    if phi.shape != k.shape:
        raise ValueError(f"phi and k must have the same shape. Got {phi.shape} vs {k.shape}.")

    mask = np.isfinite(phi) & np.isfinite(k) & (phi > 0) & (k > 0)
    if int(mask.sum()) < 10:
        raise ValueError("Too few active cells after masking. Check phi/k inputs.")

    phi_act = phi[mask]
    k_act = k[mask]

    phi_norm = np.full_like(phi, np.nan, dtype=np.float32)
    k_norm = np.full_like(k, np.nan, dtype=np.float32)

    p5 = np.percentile(phi_act, 5)
    p95 = np.percentile(phi_act, 95)
    phi_norm[mask] = (phi_act - p5) / (p95 - p5 + 1e-12)
    phi_norm[mask] = np.clip(phi_norm[mask], 0.0, 1.0)

    lk = np.log10(k_act)
    p5 = np.percentile(lk, 5)
    p95 = np.percentile(lk, 95)
    lk_norm = (lk - p5) / (p95 - p5 + 1e-12)
    lk_norm = np.clip(lk_norm, 0.0, 1.0)
    k_norm[mask] = lk_norm.astype(np.float32)

    return phi_norm, k_norm, mask

def choose_well_ij(k_norm: np.ndarray, mask: np.ndarray, mode: str, ij: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    nx, ny = k_norm.shape
    if mode == "manual":
        if ij is None:
            raise ValueError("manual mode requires ij=(i,j)")
        i, j = int(ij[0]), int(ij[1])
        i = int(np.clip(i, 0, nx - 1))
        j = int(np.clip(j, 0, ny - 1))
        if not bool(mask[i, j]):
            ii, jj = np.where(mask)
            d = (ii - i) ** 2 + (jj - j) ** 2
            k = int(np.argmin(d))
            return int(ii[k]), int(jj[k])
        return i, j

    if mode == "center":
        ii, jj = np.where(mask)
        return int(np.median(ii)), int(np.median(jj))

    if mode == "max_k":
        tmp = np.where(mask, k_norm, -np.inf)
        idx = np.unravel_index(int(np.argmax(tmp)), tmp.shape)
        return int(idx[0]), int(idx[1])

    raise ValueError("well mode must be one of: max_k, center, manual")

def _gauss2d(nx: int, ny: int, cx: int, cy: int, sigma: float) -> np.ndarray:
    x = np.arange(nx)[:, None]
    y = np.arange(ny)[None, :]
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    g = np.exp(-0.5 * r2 / (sigma ** 2 + 1e-12)).astype(np.float32)
    return g / (float(g.sum()) + 1e-12)

def _grad2d(a: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ax = np.empty_like(a, dtype=np.float32)
    ay = np.empty_like(a, dtype=np.float32)

    ax[1:-1, :] = 0.5 * (a[2:, :] - a[:-2, :])
    ax[0, :] = a[1, :] - a[0, :]
    ax[-1, :] = a[-1, :] - a[-2, :]

    ay[:, 1:-1] = 0.5 * (a[:, 2:] - a[:, :-2])
    ay[:, 0] = a[:, 1] - a[:, 0]
    ay[:, -1] = a[:, -1] - a[:, -2]
    return ax, ay

def _div2d(fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
    out = np.zeros_like(fx, dtype=np.float32)
    out[1:, :] += fx[1:, :] - fx[:-1, :]
    out[:, 1:] += fy[:, 1:] - fy[:, :-1]
    return out

def _blur2d(a: np.ndarray, mask: np.ndarray, steps: int = 1) -> np.ndarray:
    if steps <= 0:
        return a
    kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    kernel = kernel / kernel.sum()

    out = a.copy().astype(np.float32)
    for _ in range(steps):
        tmp = out.copy()
        for dx in range(-2, 3):
            tmp += kernel[dx + 2] * np.roll(out, shift=dx, axis=0)
        out = tmp / 2.0

        tmp = out.copy()
        for dy in range(-2, 3):
            tmp += kernel[dy + 2] * np.roll(out, shift=dy, axis=1)
        out = tmp / 2.0

        out = np.where(mask, out, 0.0).astype(np.float32)
    return out

def run_forward(
    phi: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    q: np.ndarray,
    params: Dict[str, float],
    well_mode: str = "max_k",
    well_ij: Optional[Tuple[int, int]] = None,
    return_pressure: bool = True,
    thr_area: float = 0.05,
) -> ForwardResult:
    params = {**DEFAULT_PARAMS, **(params or {})}

    phi_norm, k_norm, mask = prepare_phi_k(phi, k)
    nx, ny = phi.shape
    wi, wj = choose_well_ij(k_norm, mask, well_mode, ij=well_ij)

    t = np.asarray(t, dtype=np.float32).reshape(-1)
    q = np.asarray(q, dtype=np.float32).reshape(-1)
    if t.size != q.size:
        raise ValueError("t and q must have the same length.")
    Nt = int(t.size)
    if Nt < 2:
        raise ValueError("Need at least 2 timesteps.")

    src = _gauss2d(nx, ny, wi, wj, float(params["src_sigma"]))
    prod = _gauss2d(nx, ny, wi, wj, float(params["prod_sigma"]))

    dom = float(max(nx, ny))
    p_sigma = float(params["ap_diff"]) * 0.12 * dom + 1e-6

    sg = np.zeros((nx, ny), dtype=np.float32)
    sg_max = np.zeros((nx, ny), dtype=np.float32)
    sg_list: List[np.ndarray] = []
    p_list: List[np.ndarray] = []

    area = np.zeros((Nt,), dtype=np.float32)
    r_eq = np.zeros((Nt,), dtype=np.float32)

    dt = float(params["dt"])
    D0 = float(params["D0"])
    nu = float(params["nu"])
    mob_exp = float(params["mob_exp"])
    anisD = float(params["anisD"])
    Swr = float(params["Swr"])
    Sgr_max = float(params["Sgr_max"])
    C_L = float(params["C_L"])
    prod_frac = float(params["prod_frac"])
    blur_steps = int(params["blur_steps"])

    mob = np.where(mask, (k_norm ** mob_exp), 0.0).astype(np.float32)

    for n in range(Nt):
        qq = float(q[n])

        p = (float(params["alpha_p"]) * qq) * _gauss2d(nx, ny, wi, wj, p_sigma)
        p = np.where(mask, p, 0.0).astype(np.float32)

        px, py = _grad2d(p)
        vx = -mob * px
        vy = -mob * py

        sx, sy = _grad2d(sg)
        Dx = D0 * mob * anisD
        Dy = D0 * mob / max(anisD, 1e-6)
        fx = -Dx * sx
        fy = -Dy * sy
        diff_term = _div2d(fx, fy)

        adv_fx = vx * sg
        adv_fy = vy * sg
        adv_term = -_div2d(adv_fx, adv_fy)

        if qq >= 0:
            source = qq * src
            sink = 0.0
        else:
            source = 0.0
            sink = (-qq) * prod

        sg_max = np.maximum(sg_max, sg)
        sgr = np.minimum(Sgr_max, C_L * sg_max)
        mobile = np.maximum(0.0, sg - sgr)

        if qq < 0:
            mobile = np.maximum(0.0, mobile - prod_frac * sink)

        mobile_new = mobile + dt * (diff_term + nu * adv_term) + dt * source
        mobile_new = np.clip(mobile_new, 0.0, 1.0).astype(np.float32)

        sg = mobile_new + sgr
        sg = np.clip(sg, 0.0, 1.0 - Swr).astype(np.float32)

        sg = _blur2d(sg, mask, steps=blur_steps)
        sg = np.where(mask, sg, 0.0).astype(np.float32)
        sg = np.nan_to_num(sg, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        sg[sg < float(params["clip_eps"])] = 0.0

        sg_list.append(sg.copy())
        if return_pressure:
            p_list.append(p.copy())

        plume = (sg > float(thr_area)) & mask
        a = float(plume.sum())
        area[n] = a
        r_eq[n] = float(np.sqrt(a / np.pi)) if a > 0 else 0.0

    return ForwardResult(
        t=t,
        q=q,
        sg_list=sg_list,
        p_list=p_list if return_pressure else None,
        area=area,
        r_eq=r_eq,
        well_ij=(wi, wj),
    )
