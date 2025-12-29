import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from ve_core import (
    DEFAULT_PARAMS,
    prepare_phi_k,
    choose_well_ij,
    run_forward,
    box_blur_nan_safe,
)
from eclipse_io import load_eclipse_phi_k_from_uploads


st.set_page_config(page_title="VE + Darcy + Land (Universal)", layout="wide")
st.title("VE + Darcy + Land: Universal forward plume predictor")

st.markdown(
    """
Run a **paper-style VE + Darcy + Land** forward model with either:

**A) NPZ input** (`phi`, `k` arrays), or  
**B) Eclipse input** (`.DATA` + INCLUDE files or `.GRDECL/.INC` zipped).

You can either upload a **schedule CSV**, or **build a schedule** from:
- injection rate (**ton/day**),
- cycle definition (inject/soak/produce/soak),
- number of cycles and timestep (days).

The app includes:
- mask/phi/k quicklook,
- debug checks (pressure != saturation),
- optional *display-only smoothing*.
"""
)

# -----------------------------
# plotting helpers
# -----------------------------
def fig_imshow(arr, title, vmin=None, vmax=None, cmap=None):
    fig = plt.figure(figsize=(7.2, 4.4))
    plt.title(title)
    if vmin is None:
        vmin = float(np.nanpercentile(arr, 1)) if np.isfinite(arr).any() else 0.0
    if vmax is None:
        vmax = float(np.nanpercentile(arr, 99)) if np.isfinite(arr).any() else 1.0
    im = plt.imshow(arr, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap)
    plt.xlabel("j")
    plt.ylabel("i")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig

def fig_schedule(t_days, q_ton_day, tidx=None):
    fig = plt.figure(figsize=(7.2, 4.4))
    plt.title("Schedule (ton/day)")
    plt.plot(t_days, q_ton_day, linewidth=2)
    if tidx is not None and 0 <= tidx < len(t_days):
        plt.axvline(t_days[tidx], linestyle="--")
    plt.xlabel("time (days)")
    plt.ylabel("rate (ton/day), +inj / -prod")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig

def fig_timeseries(t_days, y, ylab, title):
    fig = plt.figure(figsize=(7.2, 4.4))
    plt.title(title)
    plt.plot(t_days, y, linewidth=2)
    plt.xlabel("time (days)")
    plt.ylabel(ylab)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    return fig

# -----------------------------
# schedule builder
# -----------------------------
def build_cycle_schedule_ton_day(
    dt_days: float,
    n_cycles: int,
    inj_days: float,
    soak1_days: float,
    prod_days: float,
    soak2_days: float,
    inj_rate_ton_day: float,
    prod_rate_ton_day: float,
):
    """
    Returns (t_days, q_ton_day) sampled every dt_days.
    Pattern per cycle: +inj -> 0 -> -prod -> 0
    """
    dt_days = float(dt_days)
    assert dt_days > 0

    phases = [
        (inj_days, +abs(inj_rate_ton_day)),
        (soak1_days, 0.0),
        (prod_days, -abs(prod_rate_ton_day)),
        (soak2_days, 0.0),
    ]

    t_list = []
    q_list = []
    t = 0.0

    for _ in range(int(n_cycles)):
        for dur, rate in phases:
            dur = float(dur)
            if dur <= 0:
                continue
            n = int(np.round(dur / dt_days))
            n = max(n, 1)
            for _k in range(n):
                t_list.append(t)
                q_list.append(rate)
                t += dt_days

    # include last time
    t_list.append(t)
    q_list.append(0.0)

    return np.asarray(t_list, dtype=np.float32), np.asarray(q_list, dtype=np.float32)

def load_schedule_csv(uploaded):
    df = pd.read_csv(uploaded)
    df.columns = [c.lower().strip() for c in df.columns]
    if "t" not in df.columns or "q" not in df.columns:
        raise ValueError("CSV must have columns: t, q (t in days, q in ton/day).")
    t = df["t"].to_numpy(dtype=np.float32)
    q = df["q"].to_numpy(dtype=np.float32)
    return t, q

def load_npz(uploaded):
    data = np.load(uploaded)
    if "phi" not in data or "k" not in data:
        raise ValueError("NPZ must contain keys: phi and k")
    phi = np.asarray(data["phi"], dtype=np.float32)
    k = np.asarray(data["k"], dtype=np.float32)
    return phi, k

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.header("1) Choose input type")
    input_mode = st.selectbox("Input", ["NPZ (phi/k)", "Eclipse deck / GRDECL (zip)"], index=0)

    st.divider()
    st.header("2) Upload")
    up_npz = None
    up_zip = None
    if input_mode == "NPZ (phi/k)":
        up_npz = st.file_uploader("Upload NPZ with keys phi,k", type=["npz"])
    else:
        up_zip = st.file_uploader(
            "Upload a ZIP containing .DATA and includes (or GRDECL/INC)",
            type=["zip"],
        )

    st.divider()
    st.header("3) Schedule")
    sched_mode = st.selectbox("Schedule mode", ["Build cycles (ton/day)", "Upload CSV (t,q)"], index=0)

    if sched_mode == "Upload CSV (t,q)":
        up_csv = st.file_uploader("Upload schedule CSV with columns t,q (days, ton/day)", type=["csv"])
    else:
        dt_days = st.number_input("dt (days)", value=5.0, min_value=0.1, step=0.5)
        n_cycles = st.number_input("number of cycles", value=3, min_value=1, step=1)

        inj_days = st.number_input("inject days (per cycle)", value=90.0, min_value=1.0, step=5.0)
        soak1_days = st.number_input("soak days after injection", value=30.0, min_value=0.0, step=5.0)
        prod_days = st.number_input("produce days (per cycle)", value=60.0, min_value=0.0, step=5.0)
        soak2_days = st.number_input("soak days after production", value=30.0, min_value=0.0, step=5.0)

        inj_rate = st.number_input("injection rate (ton/day)", value=1000.0, min_value=0.0, step=100.0)
        prod_rate = st.number_input("production rate (ton/day)", value=700.0, min_value=0.0, step=100.0)

    st.divider()
    st.header("4) Well placement")
    well_mode = st.selectbox("Well placement", ["max_k", "center", "manual"], index=0)
    manual_i = st.number_input("manual i", value=0, step=1)
    manual_j = st.number_input("manual j", value=0, step=1)

    st.divider()
    st.header("5) Model scaling")
    normalize_q = st.checkbox("Normalize schedule to max(|q|)=1 (recommended)", value=True)
    q_scale = st.number_input("Extra q scale factor", value=1.0, step=0.1)

    st.divider()
    st.header("6) Display")
    show_mask = st.checkbox("Show active mask", value=True)
    smooth_display = st.checkbox("Smooth display (visual only)", value=True)
    smooth_radius = st.slider("Smoothing radius", 0, 5, 2, 1)

    st.divider()
    st.header("7) Parameters (paper-ish defaults)")
    use_defaults = st.checkbox("Use DEFAULT_PARAMS", value=True)
    params = dict(DEFAULT_PARAMS)
    if not use_defaults:
        # expose a small set that matters most
        params["D0"] = float(st.number_input("D0 (diffusion base)", value=float(params["D0"]), step=0.05))
        params["src_amp"] = float(st.number_input("src_amp (source strength)", value=float(params["src_amp"]), step=0.05))
        params["anisD"] = float(st.number_input("anisD (anisotropy)", value=float(params["anisD"]), step=0.1))
        params["mob_exp"] = float(st.number_input("mob_exp (mobility exponent)", value=float(params["mob_exp"]), step=0.25))
        params["C_L"] = float(st.number_input("C_L (Land coeff)", value=float(params["C_L"]), step=0.05))
        params["Sgr_max"] = float(st.number_input("Sgr_max", value=float(params["Sgr_max"]), step=0.02))
        params["prod_frac"] = float(st.number_input("prod_frac (production removes mobile gas)", value=float(params["prod_frac"]), step=0.05))

    st.divider()
    thr_area = st.slider("Area threshold (Sg > thr)", 0.0, 0.5, 0.05, 0.01)
    run_btn = st.button("Run forward prediction", type="primary")


# -----------------------------
# Load inputs
# -----------------------------
phi = k = None

if input_mode == "NPZ (phi/k)":
    if up_npz is None:
        st.info("Upload an NPZ to start.")
        st.stop()
    try:
        phi, k = load_npz(up_npz)
    except Exception as e:
        st.error(f"Failed to read NPZ: {e}")
        st.stop()
else:
    if up_zip is None:
        st.info("Upload a ZIP with Eclipse files to start.")
        st.stop()
    try:
        phi, k = load_eclipse_phi_k_from_uploads(up_zip)
    except Exception as e:
        st.error(f"Failed to read Eclipse files: {e}")
        st.stop()

# schedule
if sched_mode == "Upload CSV (t,q)":
    if up_csv is None:
        st.info("Upload a schedule CSV to run.")
        st.stop()
    try:
        t_days, q_ton_day = load_schedule_csv(up_csv)
    except Exception as e:
        st.error(f"Failed to read schedule CSV: {e}")
        st.stop()
else:
    t_days, q_ton_day = build_cycle_schedule_ton_day(
        dt_days=dt_days,
        n_cycles=n_cycles,
        inj_days=inj_days,
        soak1_days=soak1_days,
        prod_days=prod_days,
        soak2_days=soak2_days,
        inj_rate_ton_day=inj_rate,
        prod_rate_ton_day=prod_rate,
    )

# normalize + scale
q = q_ton_day.astype(np.float32) * np.float32(q_scale)
if normalize_q:
    m = float(np.max(np.abs(q))) if q.size else 1.0
    if m > 0:
        q = (q / m).astype(np.float32)

# -----------------------------
# Quicklook inputs
# -----------------------------
colA, colB = st.columns(2)
with colA:
    st.subheader("phi")
    st.pyplot(fig_imshow(phi, f"phi | shape={phi.shape}", vmin=None, vmax=None))
with colB:
    st.subheader("k")
    st.pyplot(fig_imshow(k, f"k | shape={k.shape}", vmin=None, vmax=None))

# prep + mask
try:
    phi2, k_norm, mask = prepare_phi_k(phi, k)
except Exception as e:
    st.error(f"Invalid phi/k fields: {e}")
    st.stop()

if show_mask:
    st.subheader("Active mask (1=active)")
    st.pyplot(fig_imshow(mask.astype(np.float32), "mask", vmin=0.0, vmax=1.0))

# choose well
if well_mode == "manual":
    well_ij = (int(manual_i), int(manual_j))
else:
    well_ij = None
wi, wj = choose_well_ij(k_norm, mask, well_mode=well_mode, ij=well_ij)
st.caption(f"Selected well (i,j)=({wi},{wj}) | active={int(mask.sum())}")

# show schedule
st.subheader("Schedule")
st.pyplot(fig_schedule(t_days, q_ton_day, tidx=0))

# download the built schedule if in builder mode
if sched_mode == "Build cycles (ton/day)":
    st.download_button(
        "Download this schedule (CSV)",
        data=pd.DataFrame({"t": t_days, "q": q_ton_day}).to_csv(index=False).encode("utf-8"),
        file_name="schedule_cycles_ton_day.csv",
    )

if not run_btn:
    st.stop()

# -----------------------------
# Run model
# -----------------------------
with st.spinner("Running VE + Darcy + Land forward model..."):
    try:
        res = run_forward(
            phi=phi2,
            k=k,
            t_days=t_days,
            q_norm=q,                 # normalized schedule used for model source term
            params=params,
            well_ij=(wi, wj),
            return_pressure=True,
            thr_area=float(thr_area),
        )
    except Exception as e:
        st.error(f"Run failed: {e}")
        st.stop()

st.success("Done.")

Nt = len(res["sg_list"])
tidx = st.slider("Select timestep (tidx)", 0, max(0, Nt - 1), 0)

sg = res["sg_list"][tidx]
p = res["p_list"][tidx] if res["p_list"] is not None else None

# optional display smoothing (visual only)
sg_show = box_blur_nan_safe(sg, smooth_radius) if smooth_display else sg
p_show = box_blur_nan_safe(p, smooth_radius) if (smooth_display and p is not None) else p

left, right = st.columns(2)
with left:
    st.subheader(f"Sg predicted | tidx={tidx}")
    st.pyplot(fig_imshow(sg_show, f"Sg | tidx={tidx}", vmin=0.0, vmax=1.0))
with right:
    st.subheader(f"Pressure surrogate | tidx={tidx}")
    if p_show is None:
        st.write("Pressure output disabled.")
    else:
        # percentile scaling avoids “all purple”
        vmin = float(np.nanpercentile(p_show, 1))
        vmax = float(np.nanpercentile(p_show, 99))
        st.pyplot(fig_imshow(p_show, f"p | tidx={tidx}", vmin=vmin, vmax=vmax))

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("q(t) (ton/day)")
    st.pyplot(fig_schedule(res["t_days"], res["q_ton_day"], tidx=tidx))
with col2:
    st.subheader("Plume area")
    st.pyplot(fig_timeseries(res["t_days"], res["area"], "area (cells)", "Area time series"))
with col3:
    st.subheader("Equivalent radius")
    st.pyplot(fig_timeseries(res["t_days"], res["r_eq"], "r_eq (cells)", "Equivalent radius time series"))

# -----------------------------
# Debug panel
# -----------------------------
with st.expander("Debug (pressure != saturation)"):
    st.write("Sg stats:", float(np.nanmin(sg)), float(np.nanmax(sg)), float(np.nanmean(np.nan_to_num(sg))))
    if p is None:
        st.write("Pressure: None")
    else:
        st.write("P stats:", float(np.nanmin(p)), float(np.nanmax(p)), float(np.nanmean(np.nan_to_num(p))))
        st.write("allclose(Sg,P):", bool(np.allclose(np.nan_to_num(sg), np.nan_to_num(p))))
        st.write("same object id:", (id(sg) == id(p)))

# -----------------------------
# Download results
# -----------------------------
st.subheader("Download results")

out_npz = io.BytesIO()
sg_stack = np.stack([np.nan_to_num(s, nan=0.0) for s in res["sg_list"]], axis=0).astype(np.float32)
p_stack = None if res["p_list"] is None else np.stack([np.nan_to_num(x, nan=0.0) for x in res["p_list"]], axis=0).astype(np.float32)
np.savez_compressed(
    out_npz,
    sg=sg_stack,
    p=p_stack if p_stack is not None else np.array([], dtype=np.float32),
    t_days=res["t_days"],
    q_ton_day=res["q_ton_day"],
    area=res["area"],
    r_eq=res["r_eq"],
)
st.download_button("Download predicted fields (NPZ)", data=out_npz.getvalue(), file_name="ve_results.npz")

out_csv = pd.DataFrame(
    {"t_days": res["t_days"], "q_ton_day": res["q_ton_day"], "area": res["area"], "r_eq": res["r_eq"]}
).to_csv(index=False).encode("utf-8")
st.download_button("Download time series (CSV)", data=out_csv, file_name="plume_timeseries.csv")
