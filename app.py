import io
import zipfile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from ve_core import DEFAULT_PARAMS, run_forward, choose_well_ij, prepare_phi_k
from eclipse_io import load_eclipse_phi_k_from_upload


st.set_page_config(page_title="VE+Darcy+Land Forward Predictor", layout="wide")
st.title("VE + Darcy + Land: Forward plume prediction (robust + Eclipse-ready)")

st.markdown(
    """
Upload **phi/k** (NPZ) *or* upload an **ECLIPSE deck** (`.DATA` or `.zip` with INCLUDE files),
and an **injection schedule** (CSV) — or generate a smooth schedule in the app.

**NPZ keys**
- `phi` : 2D array (nx, ny)
- `k`   : 2D array (nx, ny)  (PERMX recommended)

**Schedule CSV columns**
- `t` : 0..Nt-1 (dense)
- `q` : signed rate (positive=injection, negative=production)

Key upgrades in this version:
- mask-aware PDE steps (no leakage into inactive cells),
- CFL-safe adaptive substepping,
- optional gentle smoothing to remove pixel noise.
"""
)

# -------------------------
# Helpers (plots)
# -------------------------
def fig_imshow(arr, title, vmin=None, vmax=None):
    fig = plt.figure(figsize=(7.2, 4.6))
    plt.title(title)
    if vmin is None:
        vmin = float(np.nanmin(arr)) if np.isfinite(arr).any() else 0.0
    if vmax is None:
        vmax = float(np.nanmax(arr)) if np.isfinite(arr).any() else 1.0
    im = plt.imshow(arr, origin="lower", vmin=vmin, vmax=vmax)
    plt.xlabel("j")
    plt.ylabel("i")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig

def fig_schedule(t, q, tidx=None):
    fig = plt.figure(figsize=(7.2, 4.6))
    plt.title("Schedule q(t)")
    plt.plot(t, q)
    if tidx is not None and 0 <= tidx < len(t):
        plt.axvline(t[tidx], linestyle="--")
    plt.xlabel("t")
    plt.ylabel("q")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def fig_timeseries(t, y, ylab, title):
    fig = plt.figure(figsize=(7.2, 4.6))
    plt.title(title)
    plt.plot(t, y)
    plt.xlabel("t")
    plt.ylabel(ylab)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def load_npz(uploaded):
    data = np.load(uploaded)
    if "phi" not in data or "k" not in data:
        raise ValueError("NPZ must contain keys: phi and k")
    return data["phi"], data["k"]

def load_schedule_csv(uploaded):
    df = pd.read_csv(uploaded)
    df.columns = [c.lower().strip() for c in df.columns]
    if "t" not in df.columns or "q" not in df.columns:
        raise ValueError("CSV must have columns: t, q")
    t = df["t"].to_numpy(dtype=np.float32)
    q = df["q"].to_numpy(dtype=np.float32)
    return t, q

def make_smooth_cycle_schedule(Nt=80, inj=1.0, prod=-0.6,
                               ramp=5, hold_inj=25, soak=10, hold_prod=20, rest=15):
    # Ensures total length Nt by trimming/padding with zeros.
    q = []
    # ramp up inject
    q += list(np.linspace(0, inj, ramp, endpoint=False))
    # hold inject
    q += [inj] * hold_inj
    # soak
    q += [0.0] * soak
    # ramp down to production
    q += list(np.linspace(0, prod, ramp, endpoint=False))
    # hold production
    q += [prod] * hold_prod
    # rest
    q += [0.0] * rest

    q = np.array(q, dtype=np.float32)
    if q.size < Nt:
        q = np.pad(q, (0, Nt - q.size), constant_values=0.0)
    else:
        q = q[:Nt]
    t = np.arange(Nt, dtype=np.float32)
    return t, q

# -------------------------
# Sidebar inputs
# -------------------------
with st.sidebar:
    st.header("1) Upload inputs")
    input_mode = st.selectbox("Input type", ["NPZ (phi/k)", "ECLIPSE deck (.DATA or .zip)"], index=0)

    up_npz = None
    up_deck = None
    if input_mode == "NPZ (phi/k)":
        up_npz = st.file_uploader("phi/k NPZ", type=["npz"])
    else:
        up_deck = st.file_uploader("ECLIPSE deck (.DATA) or ZIP with INCLUDE", type=["data", "zip", "DATA", "ZIP"])

        st.caption("ECLIPSE extract options")
        ecl_mode = st.selectbox("2D extraction", ["layer", "mean"], index=0)
        ecl_layer = st.number_input("Layer index (if layer)", value=0, step=1)
        ecl_kkey = st.text_input("Permeability keyword", value="PERMX")

    st.divider()
    st.header("2) Schedule")
    schedule_mode = st.selectbox("Schedule source", ["Upload CSV", "Generate smooth cycle"], index=1)
    up_csv = None
    if schedule_mode == "Upload CSV":
        up_csv = st.file_uploader("schedule CSV", type=["csv"])
    else:
        Nt = st.number_input("Nt (timesteps)", value=80, min_value=10, step=10)
        inj = st.number_input("Injection level (+)", value=1.0)
        prod = st.number_input("Production level (-)", value=-0.6)
        ramp = st.number_input("Ramp steps", value=5, min_value=1, step=1)
        hold_inj = st.number_input("Hold inject steps", value=25, min_value=1, step=1)
        soak = st.number_input("Soak steps", value=10, min_value=0, step=1)
        hold_prod = st.number_input("Hold produce steps", value=20, min_value=1, step=1)
        rest = st.number_input("Rest steps", value=15, min_value=0, step=1)

    st.divider()
    st.header("3) Well location")
    well_mode = st.selectbox("Well placement", ["max_k", "center", "manual"], index=0)
    manual_i = st.number_input("manual i", value=0, step=1)
    manual_j = st.number_input("manual j", value=0, step=1)

    st.divider()
    st.header("4) Stability / smoothness")
    q_scale = st.number_input("Schedule scale factor (multiplies q)", value=1.0)
    normalize_q = st.checkbox("Normalize q to max(|q|)=1", value=True)
    thr_area = st.slider("Area threshold (Sg > thr)", 0.0, 0.5, 0.05, 0.01)

    st.caption("Smoother = more substeps + small smoothing")
    cfl = st.slider("CFL (smaller = smoother, slower)", 0.1, 0.8, 0.35, 0.05)
    smooth_iters = st.slider("Extra smoothing iterations", 0, 5, 1, 1)

    st.divider()
    st.header("5) Paper parameters")
    use_defaults = st.checkbox("Use paper-calibrated defaults", value=True)
    params = dict(DEFAULT_PARAMS)
    if not use_defaults:
        for k in list(params.keys()):
            params[k] = st.number_input(k, value=float(params[k]))

    run_btn = st.button("Run forward prediction", type="primary")

# -------------------------
# Load inputs
# -------------------------
try:
    if input_mode == "NPZ (phi/k)":
        if up_npz is None:
            st.info("Upload an NPZ to begin.")
            st.stop()
        phi, k = load_npz(up_npz)
    else:
        if up_deck is None:
            st.info("Upload an ECLIPSE deck (.DATA or .zip) to begin.")
            st.stop()
        phi, k = load_eclipse_phi_k_from_upload(
            up_deck,
            mode=ecl_mode,
            layer=int(ecl_layer),
            kkey=ecl_kkey.strip().upper()
        )

    if schedule_mode == "Upload CSV":
        if up_csv is None:
            st.info("Upload a schedule CSV (t,q) or choose 'Generate smooth cycle'.")
            st.stop()
        t, q = load_schedule_csv(up_csv)
    else:
        t, q = make_smooth_cycle_schedule(
            Nt=int(Nt), inj=float(inj), prod=float(prod),
            ramp=int(ramp), hold_inj=int(hold_inj), soak=int(soak),
            hold_prod=int(hold_prod), rest=int(rest)
        )

    q = q.astype(np.float32) * np.float32(q_scale)
    if normalize_q:
        m = float(np.max(np.abs(q))) if q.size else 1.0
        if m > 0:
            q = (q / m).astype(np.float32)

except Exception as e:
    st.error(f"Failed to read inputs: {e}")
    st.stop()

# -------------------------
# Preview inputs
# -------------------------
colA, colB = st.columns(2)
with colA:
    st.subheader("Input: porosity (phi)")
    st.pyplot(fig_imshow(phi, f"phi | shape={phi.shape}"))
with colB:
    st.subheader("Input: permeability (k)")
    st.pyplot(fig_imshow(k, f"k | shape={k.shape}"))

# well selection
try:
    _, k_norm, mask = prepare_phi_k(phi, k)
    if well_mode == "manual":
        well_ij = (int(manual_i), int(manual_j))
    else:
        well_ij = None
    wi, wj = choose_well_ij(k_norm, mask, well_mode, ij=well_ij)
    st.caption(f"Selected well (i,j)=({wi},{wj}) | active cells={int(mask.sum())}")
except Exception as e:
    st.error(f"Input fields invalid: {e}")
    st.stop()

st.subheader("Injection schedule")
st.pyplot(fig_schedule(t, q, tidx=0))

# schedule download
sched_csv = pd.DataFrame({"t": t.astype(int), "q": q}).to_csv(index=False).encode("utf-8")
st.download_button("Download this schedule (CSV)", data=sched_csv, file_name="schedule_generated.csv")

if not run_btn:
    st.stop()

# -------------------------
# Run model
# -------------------------
with st.spinner("Running VE+Darcy+Land forward model..."):
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
            cfl=float(cfl),
            smooth_iters=int(smooth_iters),
        )
    except Exception as e:
        st.error(f"Run failed: {e}")
        st.stop()

st.success("Done.")

Nt_out = len(res.sg_list)
tidx = st.slider("Select timestep (tidx)", 0, max(0, Nt_out - 1), min(0, Nt_out - 1))

left, right = st.columns(2)
with left:
    st.subheader(f"Sg predicted | tidx={tidx}")
    st.pyplot(fig_imshow(res.sg_list[tidx], f"Sg predicted | tidx={tidx}", vmin=0.0, vmax=1.0))
with right:
    st.subheader(f"Pressure surrogate | tidx={tidx}")
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
    st.subheader("Plume area")
    st.pyplot(fig_timeseries(res.t, res.area, "area (cells)", "Area time series"))
with col3:
    st.subheader("Equivalent radius")
    st.pyplot(fig_timeseries(res.t, res.r_eq, "r_eq (cells)", "Equivalent radius time series"))

st.subheader("Download results")
out_npz = io.BytesIO()
sg_stack = np.stack([np.nan_to_num(s, nan=0.0) for s in res.sg_list], axis=0).astype(np.float32)
np.savez_compressed(out_npz, sg=sg_stack, t=res.t, q=res.q, area=res.area, r_eq=res.r_eq)
st.download_button("Download predicted Sg (NPZ)", data=out_npz.getvalue(), file_name="sg_predicted.npz")

out_csv = pd.DataFrame({"t": res.t, "q": res.q, "area": res.area, "r_eq": res.r_eq}).to_csv(index=False).encode("utf-8")
st.download_button("Download time series (CSV)", data=out_csv, file_name="plume_timeseries.csv")
