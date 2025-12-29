# app.py
from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from ve_core import DEFAULT_PARAMS, run_forward, choose_well_ij, prepare_phi_k

# Optional loaders
HAS_ECL = False
HAS_SPE = False
try:
    from eclipse_loader import extract_zip_to_temp, list_case_roots, load_phi_k_from_eclipse
    HAS_ECL = True
except Exception:
    HAS_ECL = False

try:
    from spe10_loader import load_spe10_zip, to_2d
    HAS_SPE = True
except Exception:
    HAS_SPE = False

st.set_page_config(page_title="Universal VE Simulator (ECLIPSE + SPE10)", layout="wide")
st.title("Universal VE Simulator: ECLIPSE grids + SPE10/SPE102 por/perm")

st.markdown(
    """
This app runs a **VE + Darcy + Land** forward simulator on:

- **NPZ**: `phi` and `k` 2D arrays  
- **ECLIPSE ZIP**: contains `*.EGRID` + `*.INIT` (PORO, PERMX, ACTNUM...)  
- **SPE10/SPE102 ZIP**: contains `spe_phi.dat` and `spe_perm.dat` (KX,KY,KZ)

Everything is converted into a **canonical 2D phi/k map**, then simulated with a schedule `t,q`.
"""
)

# -------------------------
# Helper: schedule parsing & resampling
# -------------------------
def load_schedule_csv(uploaded) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(uploaded)
    df.columns = [c.lower().strip() for c in df.columns]
    if "t" not in df.columns or "q" not in df.columns:
        raise ValueError("CSV must have columns: t, q")
    df = df[["t", "q"]].dropna().sort_values("t")
    return df["t"].to_numpy(np.float32), df["q"].to_numpy(np.float32)

def resample_schedule(t_pts: np.ndarray, q_pts: np.ndarray, Nt: int) -> Tuple[np.ndarray, np.ndarray]:
    Nt = int(max(2, Nt))
    t_grid = np.arange(Nt, dtype=np.float32)

    df = pd.DataFrame({"t": t_pts, "q": q_pts}).groupby("t", as_index=False).last()
    t_pts = df["t"].to_numpy(np.float32)
    q_pts = df["q"].to_numpy(np.float32)

    q_grid = np.interp(t_grid, t_pts, q_pts, left=q_pts[0], right=q_pts[-1]).astype(np.float32)
    return t_grid, q_grid

def fig_imshow(arr, title, vmin=None, vmax=None):
    fig = plt.figure(figsize=(7.2, 4.6))
    plt.title(title)
    if vmin is None: vmin = float(np.nanmin(arr))
    if vmax is None: vmax = float(np.nanmax(arr))
    im = plt.imshow(arr, origin="lower", vmin=vmin, vmax=vmax)
    plt.xlabel("j")
    plt.ylabel("i")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig

def fig_schedule(t, q, tidx=None):
    fig = plt.figure(figsize=(7.2, 4.6))
    plt.title("Schedule q(t) (resampled)")
    plt.plot(t, q)
    if tidx is not None and 0 <= tidx < len(t):
        plt.axvline(t[tidx], linestyle="--")
    plt.xlabel("timestep")
    plt.ylabel("q")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def fig_timeseries(t, y, ylab, title):
    fig = plt.figure(figsize=(7.2, 4.6))
    plt.title(title)
    plt.plot(t, y)
    plt.xlabel("timestep")
    plt.ylabel(ylab)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("1) Input type")
    modes = ["NPZ (phi/k)"]
    if HAS_ECL:
        modes.append("ECLIPSE ZIP (EGRID+INIT)")
    if HAS_SPE:
        modes.append("SPE10/SPE102 ZIP (spe_phi.dat + spe_perm.dat)")
    input_mode = st.selectbox("Choose input mode", modes, index=0)

    st.divider()
    st.header("2) Upload files")
    up_npz = up_zip = up_spe = None
    if input_mode.startswith("NPZ"):
        up_npz = st.file_uploader("phi/k NPZ", type=["npz"])
    elif input_mode.startswith("ECLIPSE"):
        up_zip = st.file_uploader("ECLIPSE ZIP", type=["zip"])
    else:
        up_spe = st.file_uploader("SPE10/SPE102 ZIP", type=["zip"])

    up_csv = st.file_uploader("Schedule CSV (t,q)", type=["csv"])

    st.divider()
    st.header("3) 3D -> 2D mapping")
    map_mode = st.selectbox("Map mode", ["layer", "mean"], index=0)
    layer = st.number_input("layer index (if layer mode)", value=0, min_value=0, step=1)

    st.divider()
    st.header("4) Schedule shaping")
    Nt = st.number_input("Number of timesteps (Nt)", value=80, min_value=2, step=1)
    q_scale = st.number_input("q scale", value=1.0)
    normalize_q = st.checkbox("Normalize to max(|q|)=1", value=True)

    st.divider()
    st.header("5) Well")
    well_mode = st.selectbox("Well placement", ["max_k", "center", "manual"], index=0)
    manual_i = st.number_input("manual i", value=0, step=1)
    manual_j = st.number_input("manual j", value=0, step=1)

    st.divider()
    st.header("6) Model")
    thr_area = st.slider("Area threshold Sg>thr", 0.0, 0.5, 0.05, 0.01)
    use_defaults = st.checkbox("Use defaults", value=True)
    params = dict(DEFAULT_PARAMS)
    if not use_defaults:
        for k in list(params.keys()):
            params[k] = st.number_input(k, value=float(params[k]))

    st.divider()
    run_btn = st.button("Run simulation", type="primary")

# -------------------------
# Load schedule
# -------------------------
if up_csv is None:
    st.info("Upload a schedule CSV to begin.")
    st.stop()

try:
    t_pts, q_pts = load_schedule_csv(up_csv)
    t, q = resample_schedule(t_pts, q_pts, int(Nt))
    q = q.astype(np.float32) * float(q_scale)
    if normalize_q:
        m = float(np.max(np.abs(q))) if q.size else 1.0
        if m > 0:
            q = (q / m).astype(np.float32)
except Exception as e:
    st.error(f"Schedule load failed: {e}")
    st.stop()

# -------------------------
# Load phi/k
# -------------------------
phi = k = None

if input_mode.startswith("NPZ"):
    if up_npz is None:
        st.info("Upload a phi/k NPZ.")
        st.stop()
    try:
        data = np.load(up_npz)
        phi = data["phi"].astype(np.float32)
        k = data["k"].astype(np.float32)
    except Exception as e:
        st.error(f"NPZ load failed: {e}")
        st.stop()

elif input_mode.startswith("ECLIPSE"):
    if up_zip is None:
        st.info("Upload an ECLIPSE ZIP.")
        st.stop()
    try:
        tmp = extract_zip_to_temp(up_zip.getvalue())
        roots = list_case_roots(tmp)
        if not roots:
            raise ValueError("No *.EGRID found in zip.")
        root = st.selectbox("Detected cases", roots, index=0)
        phi, k, _ = load_phi_k_from_eclipse(tmp, root=root, kkey="PERMX", mode=map_mode, layer=int(layer))
    except Exception as e:
        st.error(f"ECLIPSE load failed: {e}")
        st.stop()

else:
    if up_spe is None:
        st.info("Upload an SPE10/SPE102 ZIP.")
        st.stop()
    try:
        props = load_spe10_zip(up_spe.getvalue())
        kcomp = st.selectbox("Permeability component", ["KX", "KY", "KZ"], index=0)
        if kcomp == "KX":
            k3 = props.kx
        elif kcomp == "KY":
            k3 = props.ky
        else:
            k3 = props.kz
        phi = to_2d(props.phi, mode=map_mode, layer=int(layer))
        k = to_2d(k3, mode=map_mode, layer=int(layer))
    except Exception as e:
        st.error(f"SPE load failed: {e}")
        st.stop()

# -------------------------
# Preview + well
# -------------------------
colA, colB = st.columns(2)
with colA:
    st.subheader("phi")
    st.pyplot(fig_imshow(phi, f"phi | shape={phi.shape}"))
with colB:
    st.subheader("k")
    st.pyplot(fig_imshow(k, f"k | shape={k.shape}"))

try:
    _, k_norm, mask = prepare_phi_k(phi, k)
    well_ij = (int(manual_i), int(manual_j)) if well_mode == "manual" else None
    wi, wj = choose_well_ij(k_norm, mask, well_mode, ij=well_ij)
    st.caption(f"Selected well (i,j)=({wi},{wj}) | active cells={int(mask.sum())}")
except Exception as e:
    st.error(f"Input invalid: {e}")
    st.stop()

st.subheader("Schedule")
st.pyplot(fig_schedule(t, q, tidx=0))

phi_k_npz = io.BytesIO()
np.savez_compressed(phi_k_npz, phi=np.asarray(phi, np.float32), k=np.asarray(k, np.float32))
st.download_button("Download phi/k as NPZ", data=phi_k_npz.getvalue(), file_name="phi_k_export.npz")

if not run_btn:
    st.stop()

# -------------------------
# Run
# -------------------------
with st.spinner("Running..."):
    try:
        res = run_forward(
            phi=phi,
            k=k,
            t=t,
            q=q,
            params=params,
            well_mode=well_mode,
            well_ij=(int(manual_i), int(manual_j)) if well_mode == "manual" else None,
            return_pressure=True,
            thr_area=float(thr_area),
        )
    except Exception as e:
        st.error(f"Run failed: {e}")
        st.stop()

st.success("Done.")

Nt2 = len(res.sg_list)
tidx = st.slider("tidx", 0, max(0, Nt2 - 1), 0)

left, right = st.columns(2)
with left:
    st.subheader(f"Sg | tidx={tidx}")
    st.pyplot(fig_imshow(res.sg_list[tidx], f"Sg predicted | tidx={tidx}", vmin=0.0, vmax=1.0))
with right:
    st.subheader(f"Pressure | tidx={tidx}")
    p = res.p_list[tidx] if res.p_list is not None else None
    if p is None:
        st.write("Pressure output disabled.")
    else:
        st.pyplot(fig_imshow(p, f"p predicted | tidx={tidx}"))

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("q(t)")
    st.pyplot(fig_schedule(res.t, res.q, tidx=tidx))
with col2:
    st.subheader("Area")
    st.pyplot(fig_timeseries(res.t, res.area, "area (cells)", "Area time series"))
with col3:
    st.subheader("Equivalent radius")
    st.pyplot(fig_timeseries(res.t, res.r_eq, "r_eq (cells)", "Equivalent radius time series"))

st.subheader("Download results")
out_npz = io.BytesIO()
sg_stack = np.stack([np.nan_to_num(s, nan=0.0) for s in res.sg_list], axis=0).astype(np.float32)
np.savez_compressed(out_npz, sg=sg_stack, t=res.t, q=res.q, area=res.area, r_eq=res.r_eq, well=np.array(res.well_ij))
st.download_button("Download predicted Sg (NPZ)", data=out_npz.getvalue(), file_name="sg_predicted.npz")

out_csv = pd.DataFrame({"t": res.t, "q": res.q, "area": res.area, "r_eq": res.r_eq}).to_csv(index=False).encode("utf-8")
st.download_button("Download time series (CSV)", data=out_csv, file_name="plume_timeseries.csv")
