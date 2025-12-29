from dataclasses import dataclass
import numpy as np


DEFAULT_PARAMS = dict(
    # transport / spreading
    D0=0.25,
    alpha_p=1.0,
    src_amp=0.40,
    prod_frac=0.60,
    Swr=0.20,
    Sgr_max=0.35,
    C_L=2.0,
    hc=0.05,
    mob_exp=1.6,
    anisD=1.0,
    eps_h=1e-3,
    nu=0.02,
    m_spread=1.4,
    ap_diff=1.0,
    qp_amp=1.0,
    # well footprint
    rad_w=2.5,
)

@dataclass
class ForwardResult:
    t: np.ndarray
    q: np.ndarray
    sg_list: list
    p_list: list | None
    area: np.ndarray
    r_eq: np.ndarray


# -------------------------
# Utilities: masking + plotting support
# -------------------------
def prepare_phi_k(phi, k):
    phi = np.array(phi, dtype=np.float32)
    k = np.array(k, dtype=np.float32)
    if phi.ndim != 2 or k.ndim != 2:
        raise ValueError("phi and k must be 2D arrays")
    if phi.shape != k.shape:
        raise ValueError("phi and k must have the same shape")

    mask = np.isfinite(phi) & np.isfinite(k) & (phi > 0) & (k > 0)
    if mask.sum() < 10:
        raise ValueError("Too few active cells (check ACTNUM/masking or layer selection).")

    # normalize permeability to [0,1] on active cells
    kk = np.where(mask, k, np.nan)
    lo = np.nanpercentile(kk, 5)
    hi = np.nanpercentile(kk, 95)
    hi = max(hi, lo + 1e-12)
    k_norm = np.clip((k - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    k_norm = np.where(mask, k_norm, 0.0).astype(np.float32)

    phi_m = np.where(mask, phi, np.nan).astype(np.float32)
    return phi_m, k_norm, mask


def choose_well_ij(k_norm, mask, mode="max_k", ij=None):
    nx, ny = k_norm.shape
    if mode == "center":
        wi, wj = nx // 2, ny // 2
        if not mask[wi, wj]:
            # nearest active
            idx = np.argwhere(mask)
            d = np.sum((idx - np.array([wi, wj])) ** 2, axis=1)
            wi, wj = idx[int(np.argmin(d))]
        return int(wi), int(wj)

    if mode == "manual":
        if ij is None:
            raise ValueError("manual mode requires ij=(i,j)")
        wi, wj = int(ij[0]), int(ij[1])
        wi = int(np.clip(wi, 0, nx - 1))
        wj = int(np.clip(wj, 0, ny - 1))
        if not mask[wi, wj]:
            raise ValueError("Manual (i,j) is not active (ACTNUM=0 or invalid).")
        return wi, wj

    # max_k
    kk = np.where(mask, k_norm, -1.0)
    wi, wj = np.unravel_index(int(np.argmax(kk)), kk.shape)
    return int(wi), int(wj)


# -------------------------
# Core numerics (mask-aware, CFL-safe)
# -------------------------
def binomial_blur2d(a, mask, iters=1):
    # Kernel: [[1,2,1],[2,4,2],[1,2,1]] / 16
    if iters <= 0:
        return a
    out = a.copy()
    for _ in range(iters):
        c = out
        # pad with edge values
        p = np.pad(c, ((1, 1), (1, 1)), mode="edge")
        b = (
            1*p[0:-2,0:-2] + 2*p[0:-2,1:-1] + 1*p[0:-2,2:] +
            2*p[1:-1,0:-2] + 4*p[1:-1,1:-1] + 2*p[1:-1,2:] +
            1*p[2:,0:-2] + 2*p[2:,1:-1] + 1*p[2:,2:]
        ) / 16.0
        out = np.where(mask, b, 0.0).astype(np.float32)
    return out


def laplacian_masked(a, mask):
    # no-flow across inactive boundaries: neighbor inactive => use center value
    c = a
    nx, ny = c.shape
    out = np.zeros_like(c, dtype=np.float32)
    # neighbors
    for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
        ni = np.clip(np.arange(nx)[:, None] + di, 0, nx-1)
        nj = np.clip(np.arange(ny)[None, :] + dj, 0, ny-1)
        neigh = c[ni, nj]
        neigh_mask = mask[ni, nj]
        neigh = np.where(neigh_mask, neigh, c)  # enforce zero gradient
        out += (neigh - c)
    out = np.where(mask, out, 0.0).astype(np.float32)
    return out


def upwind_advect_masked(h, ux, uy, dt, mask):
    # Simple first-order upwind, with boundary protection on inactive cells.
    nx, ny = h.shape
    h0 = h

    # x-direction (i)
    hx_m1 = np.roll(h0, 1, axis=0)
    hx_p1 = np.roll(h0, -1, axis=0)
    mx_m1 = np.roll(mask, 1, axis=0)
    mx_p1 = np.roll(mask, -1, axis=0)
    hx_m1 = np.where(mx_m1, hx_m1, h0)
    hx_p1 = np.where(mx_p1, hx_p1, h0)

    # y-direction (j)
    hy_m1 = np.roll(h0, 1, axis=1)
    hy_p1 = np.roll(h0, -1, axis=1)
    my_m1 = np.roll(mask, 1, axis=1)
    my_p1 = np.roll(mask, -1, axis=1)
    hy_m1 = np.where(my_m1, hy_m1, h0)
    hy_p1 = np.where(my_p1, hy_p1, h0)

    # upwind differencing
    dhdx = np.where(ux >= 0, (h0 - hx_m1), (hx_p1 - h0))
    dhdy = np.where(uy >= 0, (h0 - hy_m1), (hy_p1 - h0))

    h1 = h0 - dt * (ux * dhdx + uy * dhdy)
    h1 = np.where(mask, h1, 0.0)
    return h1.astype(np.float32)


def gaussian_pressure(nx, ny, wi, wj, amp=1.0, sigma=3.0, mask=None):
    ii = np.arange(nx)[:, None]
    jj = np.arange(ny)[None, :]
    r2 = (ii - wi) ** 2 + (jj - wj) ** 2
    p = amp * np.exp(-0.5 * r2 / (sigma * sigma))
    if mask is not None:
        p = np.where(mask, p, 0.0)
    return p.astype(np.float32)


def velocity_from_pressure(p, k_norm, alpha_p=1.0, mask=None):
    # Darcy-like velocity proportional to -k * grad(p)
    # central difference with edge padding
    pp = np.pad(p, ((1, 1), (1, 1)), mode="edge")
    dpdi = 0.5 * (pp[2:, 1:-1] - pp[:-2, 1:-1])
    dpdj = 0.5 * (pp[1:-1, 2:] - pp[1:-1, :-2])
    ux = -alpha_p * k_norm * dpdi
    uy = -alpha_p * k_norm * dpdj
    if mask is not None:
        ux = np.where(mask, ux, 0.0)
        uy = np.where(mask, uy, 0.0)
    return ux.astype(np.float32), uy.astype(np.float32)


def k_spreading_power_aniso(h, k_norm, D0x, D0y, eps_h, m_spread, dt, mask):
    # nonlinear spreading coefficient D ~ D0 * (k+eps)^m
    Dkx = D0x * np.power(k_norm + eps_h, m_spread)
    Dky = D0y * np.power(k_norm + eps_h, m_spread)
    # diffusion-like term: div(D grad h) approx D * laplacian(h)
    # (simple but stable when substepped)
    lap = laplacian_masked(h, mask)
    out = h + dt * (Dkx + Dky) * lap * 0.25
    return np.where(mask, out, 0.0).astype(np.float32)


def apply_well_source_sink(h, q_sign, q_w, src_amp, prod_frac, wi, wj, rad_w, dt, mask):
    nx, ny = h.shape
    ii = np.arange(nx)[:, None]
    jj = np.arange(ny)[None, :]
    r2 = (ii - wi) ** 2 + (jj - wj) ** 2
    kernel = np.exp(-0.5 * r2 / (rad_w * rad_w)).astype(np.float32)
    kernel = np.where(mask, kernel, 0.0)
    s = kernel / (kernel.sum() + 1e-12)

    if q_sign >= 0:
        dh = (src_amp * q_w) * s
    else:
        dh = (-prod_frac * abs(q_w)) * s

    out = h + dt * dh
    return np.where(mask, out, 0.0).astype(np.float32)


def ve_mobile_sg_from_h(h, Swr=0.2, hc=0.05, mob_exp=1.6):
    # smooth monotone mapping of thickness -> mobile saturation
    # h in [0,1]; below hc, little mobile gas; above hc, ramps up.
    x = np.clip((h - hc) / max(1e-6, (1.0 - hc)), 0.0, 1.0)
    sg = np.power(x, mob_exp)
    # enforce connate water cutoff
    sg = sg * (1.0 - Swr)
    return np.clip(sg, 0.0, 1.0).astype(np.float32)


def land_residual(Sg_max_hist, Sgr_max=0.35, C_L=2.0):
    # A simple Land-type residual as function of maximum historical saturation
    # bounded by Sgr_max
    x = np.clip(Sg_max_hist, 0.0, 1.0)
    sgr = (Sgr_max * x) / (1.0 + C_L * x + 1e-12)
    return np.clip(sgr, 0.0, Sgr_max).astype(np.float32)


def equivalent_radius(area_cells):
    return np.sqrt(area_cells / np.pi).astype(np.float32)


def run_forward(
    phi,
    k,
    t,
    q,
    params,
    well_mode="max_k",
    well_ij=None,
    return_pressure=True,
    thr_area=0.05,
    cfl=0.35,
    smooth_iters=1,
):
    phi_m, k_norm, mask = prepare_phi_k(phi, k)
    wi, wj = choose_well_ij(k_norm, mask, mode=well_mode, ij=well_ij)

    t = np.array(t, dtype=np.float32)
    q = np.array(q, dtype=np.float32)
    if t.ndim != 1 or q.ndim != 1 or len(t) != len(q):
        raise ValueError("t and q must be 1D arrays with the same length")
    Nt = len(t)
    if Nt < 2:
        raise ValueError("Need at least 2 timesteps")

    # unpack params
    D0 = float(params["D0"])
    alpha_p = float(params["alpha_p"])
    src_amp = float(params["src_amp"])
    prod_frac = float(params["prod_frac"])
    Swr = float(params["Swr"])
    Sgr_max = float(params["Sgr_max"])
    C_L = float(params["C_L"])
    hc = float(params["hc"])
    mob_exp = float(params["mob_exp"])
    anisD = float(params["anisD"])
    eps_h = float(params["eps_h"])
    nu = float(params["nu"])
    m_spread = float(params["m_spread"])
    qp_amp = float(params["qp_amp"])
    rad_w = float(params.get("rad_w", 2.5))

    # Initial conditions
    h = np.zeros_like(k_norm, dtype=np.float32)
    Sg_max_hist = np.zeros_like(k_norm, dtype=np.float32)

    sg_list = []
    p_list = [] if return_pressure else None
    area = np.zeros(Nt, dtype=np.float32)
    r_eq = np.zeros(Nt, dtype=np.float32)

    # fixed dt based on index spacing if t is dense; otherwise use dt=1
    # (Streamlit schedules should be dense 0..Nt-1)
    dt_global = 1.0

    for n in range(Nt):
        qt = float(q[n]) * qp_amp
        q_sign = 1.0 if qt >= 0 else -1.0
        q_w = abs(qt)

        # pressure surrogate (Gaussian)
        # sigma can be tuned; we tie it weakly to well radius for smoothness
        sigma = max(2.5, rad_w * 1.5)
        p = gaussian_pressure(h.shape[0], h.shape[1], wi, wj, amp=q_w, sigma=sigma, mask=mask)

        # velocities
        ux, uy = velocity_from_pressure(p, k_norm, alpha_p=alpha_p, mask=mask)

        # CFL-safe substepping
        umax = float(np.max(np.abs(ux[mask]))) if mask.any() else 0.0
        vmax = float(np.max(np.abs(uy[mask]))) if mask.any() else 0.0
        v = max(umax, vmax, 1e-8)
        dt_cfl = cfl / v
        nsub = int(np.ceil(dt_global / dt_cfl))
        nsub = max(1, min(nsub, 200))
        dt = dt_global / nsub

        D0x = D0
        D0y = D0 * anisD

        for _ in range(nsub):
            h = upwind_advect_masked(h, ux, uy, dt=dt, mask=mask)
            h = k_spreading_power_aniso(h, k_norm, D0x=D0x, D0y=D0y, eps_h=eps_h, m_spread=m_spread, dt=dt, mask=mask)
            h = h + (nu * dt) * laplacian_masked(h, mask)
            h = apply_well_source_sink(h, q_sign, q_w, src_amp, prod_frac, wi, wj, rad_w, dt=dt, mask=mask)
            h = np.clip(h, 0.0, 1.0).astype(np.float32)

            if smooth_iters > 0:
                h = binomial_blur2d(h, mask, iters=smooth_iters)

        sg_mob = ve_mobile_sg_from_h(h, Swr=Swr, hc=hc, mob_exp=mob_exp)
        Sg_max_hist = np.maximum(Sg_max_hist, sg_mob)
        sg_res = land_residual(Sg_max_hist, Sgr_max=Sgr_max, C_L=C_L)
        sg_tot = np.maximum(sg_mob, sg_res)
        sg_tot = np.where(mask, sg_tot, np.nan).astype(np.float32)

        sg_list.append(sg_tot)
        if return_pressure:
            p_list.append(np.where(mask, p, np.nan).astype(np.float32))

        # metrics
        plume = np.isfinite(sg_tot) & (sg_tot > thr_area)
        a = float(plume.sum())
        area[n] = a
        r_eq[n] = float(equivalent_radius(np.array(a, dtype=np.float32)))

    return ForwardResult(
        t=t.astype(np.float32),
        q=q.astype(np.float32),
        sg_list=sg_list,
        p_list=p_list,
        area=area,
        r_eq=r_eq,
    )
