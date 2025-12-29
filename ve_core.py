import numpy as np

DEFAULT_PARAMS = {
    # diffusion-ish spreading
    "D0": 0.25,
    "anisD": 1.0,       # >1 spreads more in x (j), <1 more in y (i)
    "n_sub": 3,         # substeps per timestep for stability
    "dt_ref": 1.0,      # dt scaling reference (days)

    # source / sink
    "src_amp": 0.65,    # how strong normalized q drives saturation near well
    "src_sigma": 2.0,   # gaussian radius for injection footprint (cells)
    "prod_frac": 0.6,   # production removes mobile gas fraction

    # mobility / nonlinearity
    "mob_exp": 2.0,     # higher -> sharper front

    # Land trapping (simple, stable hysteresis proxy)
    "C_L": 0.25,
    "Sgr_max": 0.35,

    # small floor to avoid divide-by-zero
    "eps": 1e-8,
}


def prepare_phi_k(phi, k):
    """
    Returns:
      phi2 : float32, inactive->nan
      k_norm : normalized log10(k) in [0,1] over active region
      mask : active boolean
    """
    phi = np.asarray(phi, dtype=np.float32)
    k = np.asarray(k, dtype=np.float32)
    if phi.ndim != 2 or k.ndim != 2:
        raise ValueError("phi and k must be 2D arrays.")
    if phi.shape != k.shape:
        raise ValueError(f"phi shape {phi.shape} != k shape {k.shape}")

    mask = np.isfinite(phi) & np.isfinite(k) & (phi > 0) & (k > 0)
    if not mask.any():
        raise ValueError("No active cells (phi/k non-finite or <=0 everywhere).")

    # normalize permeability using log10
    lk = np.full_like(k, np.nan, dtype=np.float32)
    lk[mask] = np.log10(k[mask].astype(np.float32))
    lo = float(np.nanpercentile(lk[mask], 1))
    hi = float(np.nanpercentile(lk[mask], 99))
    hi = max(hi, lo + 1e-6)
    k_norm = (lk - lo) / (hi - lo)
    k_norm = np.clip(k_norm, 0.0, 1.0).astype(np.float32)

    phi2 = phi.copy()
    phi2[~mask] = np.nan
    return phi2, k_norm, mask


def choose_well_ij(k_norm, mask, well_mode="max_k", ij=None):
    """
    well_mode: max_k | center | manual
    """
    ny, nx = mask.shape[1], mask.shape[0]  # not used; keep consistent
    if well_mode == "manual":
        if ij is None:
            raise ValueError("manual mode requires ij=(i,j)")
        i, j = int(ij[0]), int(ij[1])
        if i < 0 or j < 0 or i >= mask.shape[0] or j >= mask.shape[1]:
            raise ValueError("manual well ij out of bounds.")
        if not mask[i, j]:
            # move to nearest active if manual picks inactive
            coords = np.argwhere(mask)
            d = np.sum((coords - np.array([i, j])) ** 2, axis=1)
            i2, j2 = coords[int(np.argmin(d))]
            return int(i2), int(j2)
        return i, j

    if well_mode == "center":
        coords = np.argwhere(mask)
        ci = int(np.round(coords[:, 0].mean()))
        cj = int(np.round(coords[:, 1].mean()))
        if mask[ci, cj]:
            return ci, cj
        # nearest active
        d = np.sum((coords - np.array([ci, cj])) ** 2, axis=1)
        i2, j2 = coords[int(np.argmin(d))]
        return int(i2), int(j2)

    # max_k
    kk = np.where(mask, k_norm, -np.inf)
    idx = np.unravel_index(int(np.argmax(kk)), kk.shape)
    return int(idx[0]), int(idx[1])


def _laplacian(a):
    """
    5-point laplacian with edge padding behavior via roll.
    """
    return (
        -4.0 * a
        + np.roll(a, 1, axis=0)
        + np.roll(a, -1, axis=0)
        + np.roll(a, 1, axis=1)
        + np.roll(a, -1, axis=1)
    )


def _gaussian_source(shape, wi, wj, sigma):
    """
    Gaussian centered at well. Returns float32 array.
    """
    ii = np.arange(shape[0], dtype=np.float32)[:, None]
    jj = np.arange(shape[1], dtype=np.float32)[None, :]
    r2 = (ii - wi) ** 2 + (jj - wj) ** 2
    g = np.exp(-0.5 * r2 / (sigma * sigma)).astype(np.float32)
    g /= max(float(g.sum()), 1e-8)
    return g


def _land_residual_from_sgmax(sg_max, C_L, Sgr_max):
    # stable monotonic residual saturation proxy
    # Sgr increases with max historical gas saturation
    return (Sgr_max * (1.0 - np.exp(-C_L * np.clip(sg_max, 0.0, 1.0)))).astype(np.float32)


def run_forward(
    phi,
    k,
    t_days,
    q_norm,
    params,
    well_ij,
    return_pressure=True,
    thr_area=0.05,
):
    """
    Forward VE-ish model:
    - sg evolves with diffusion-like spreading weighted by k_norm and nonlinear mobility
    - injection adds sg near well (gaussian source)
    - production removes mobile gas but keeps Land residual trapped component
    - pressure is a separate, smooth field (not sg), built from a Green-function-like kernel

    Inputs:
      t_days: time array in days (monotonic)
      q_norm: normalized schedule same length as t_days (or t_days[:-1] also ok; we handle)
    """
    p = dict(DEFAULT_PARAMS)
    p.update(dict(params or {}))

    phi2, k_norm, mask = prepare_phi_k(phi, k)

    t_days = np.asarray(t_days, dtype=np.float32).reshape(-1)
    q_norm = np.asarray(q_norm, dtype=np.float32).reshape(-1)
    if len(q_norm) == len(t_days) - 1:
        # interpret as per-interval
        q_norm = np.concatenate([q_norm, q_norm[-1:]], axis=0)
    if len(q_norm) != len(t_days):
        raise ValueError("q_norm length must match t_days (or be len(t_days)-1).")

    wi, wj = int(well_ij[0]), int(well_ij[1])

    sg = np.zeros_like(phi2, dtype=np.float32)
    sg[~mask] = np.nan
    sg_max = np.zeros_like(phi2, dtype=np.float32)  # store historical max sg
    sg_max[~mask] = 0.0

    # precompute injection footprint
    gsrc = _gaussian_source(phi2.shape, wi, wj, sigma=float(p["src_sigma"]))
    gsrc = np.where(mask, gsrc, 0.0).astype(np.float32)

    sg_list = []
    p_list = [] if return_pressure else None
    area = []
    r_eq = []

    # for pressure: precompute a smoother kernel
    gP = _gaussian_source(phi2.shape, wi, wj, sigma=max(3.0, float(p["src_sigma"]) * 2.5))
    gP = np.where(mask, gP, 0.0).astype(np.float32)

    # simulate
    for n in range(len(t_days)):
        sg_list.append(sg.copy())

        if return_pressure:
            # pressure is NOT saturation: smooth response + lower in high-k regions
            # normalized for display and stable across datasets
            invk = np.where(mask, 1.0 / (k_norm + float(p["eps"])), 0.0).astype(np.float32)
            pfield = (abs(float(q_norm[n])) * gP) * (0.35 * invk + 0.65)
            pfield = pfield.astype(np.float32)
            pfield[~mask] = np.nan
            p_list.append(pfield.copy())

        # last step stores state only
        if n == len(t_days) - 1:
            break

        dt = float(t_days[n + 1] - t_days[n])
        dt = max(dt, 1e-6)

        qn = float(q_norm[n])

        # compute residual saturation from Land
        sgr = _land_residual_from_sgmax(sg_max, float(p["C_L"]), float(p["Sgr_max"]))

        # split mobile / trapped
        sg_mobile = np.clip(sg - sgr, 0.0, 1.0).astype(np.float32)

        # injection adds mobile gas near well
        if qn > 0:
            add = (float(p["src_amp"]) * qn) * gsrc
            sg_mobile = np.clip(sg_mobile + add, 0.0, 1.0).astype(np.float32)

        # production removes mobile gas, keeps trapped residual
        if qn < 0:
            rem = float(p["prod_frac"]) * abs(qn)
            sg_mobile = np.clip(sg_mobile * (1.0 - rem), 0.0, 1.0).astype(np.float32)

        # nonlinear mobility: sharper fronts
        mob = (sg_mobile ** float(p["mob_exp"])).astype(np.float32)

        # diffusion-like spreading, weighted by permeability
        D0 = float(p["D0"])
        anis = float(p["anisD"])
        n_sub = int(p["n_sub"])

        # stable substepping
        subdt = dt / max(n_sub, 1)
        for _ in range(max(n_sub, 1)):
            # anisotropic diffusion via directional scaling
            lap = _laplacian(mob)
            # axis0 = i, axis1 = j
            lap_i = (
                -2.0 * mob
                + np.roll(mob, 1, axis=0)
                + np.roll(mob, -1, axis=0)
            )
            lap_j = (
                -2.0 * mob
                + np.roll(mob, 1, axis=1)
                + np.roll(mob, -1, axis=1)
            )
            lap_anis = (lap_i + (anis * anis) * lap_j).astype(np.float32)

            # local diffusion coefficient from k_norm
            Dloc = (D0 * (0.25 + 0.75 * k_norm)).astype(np.float32)
            mob = mob + subdt / float(p["dt_ref"]) * (Dloc * lap_anis)
            mob = np.clip(mob, 0.0, 1.0).astype(np.float32)
            mob[~mask] = 0.0

        # invert mobility back to saturation space
        sg_mobile = np.clip(mob ** (1.0 / max(float(p["mob_exp"]), 1e-6)), 0.0, 1.0).astype(np.float32)

        # total gas saturation = residual + mobile
        sg = np.clip(sgr + sg_mobile, 0.0, 1.0).astype(np.float32)
        sg[~mask] = np.nan

        # update historical max
        sg_max = np.maximum(sg_max, np.nan_to_num(sg, nan=0.0)).astype(np.float32)

        # area + equivalent radius
        a = float(np.sum((sg > float(thr_area)) & mask))
        area.append(a)
        r_eq.append(float(np.sqrt(a / np.pi)) if a > 0 else 0.0)

    # finalize time series lengths
    area = np.asarray(area + [area[-1] if len(area) else 0.0], dtype=np.float32)
    r_eq = np.asarray(r_eq + [r_eq[-1] if len(r_eq) else 0.0], dtype=np.float32)

    return {
        "sg_list": sg_list,
        "p_list": p_list,
        "t_days": t_days,
        "q_ton_day": (q_norm * 0.0).astype(np.float32),  # app plots original ton/day; kept placeholder
        "area": area,
        "r_eq": r_eq,
        "mask": mask,
        "k_norm": k_norm,
    }


def box_blur_nan_safe(a, r=2):
    """
    Simple NaN-safe box blur for display-only smoothing (no scipy needed).
    """
    if a is None:
        return None
    if r <= 0:
        return a
    a = np.asarray(a, dtype=np.float32)
    valid = np.isfinite(a).astype(np.float32)
    x = np.nan_to_num(a, nan=0.0).astype(np.float32)

    k = 2 * r + 1

    def conv2(z):
        zp = np.pad(z, ((r, r), (r, r)), mode="edge")
        out = np.zeros_like(z, dtype=np.float32)
        for di in range(k):
            for dj in range(k):
                out += zp[di : di + z.shape[0], dj : dj + z.shape[1]]
        return out

    num = conv2(x)
    den = conv2(valid)
    out = np.where(den > 1e-8, num / den, np.nan).astype(np.float32)
    out[~np.isfinite(a)] = np.nan
    return out
