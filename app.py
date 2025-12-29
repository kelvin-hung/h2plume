import os
import sys
from pathlib import Path

# Ensure local imports work on Streamlit Cloud
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Local modules (safe import)
from ve_core import DEFAULT_PARAMS, run_forward, choose_well_ij, prepare_phi_k
from schedule_tools import build_cycle_schedule_ton_per_day, moving_average
from eclipse_io import load_eclipse_phi_k_from_uploads

st.set_page_config(page_title="VE+Darcy+Land Forward Predictor", layout="wide")
st.title("VE + Darcy + Land: Forward plume prediction (ton/day cyclic scheduling)")

st.markdown(
    """
This app runs a forward plume predictor from **phi/k** and an **injection schedule**.

**Inputs supported**
- **NPZ**: keys `phi` and `k` (2D arrays)
- **Eclipse TEXT deck**: upload main `.DATA` and also upload **all INCLUDE files** (`.INC/.GRDECL`) via the second uploader.

**Schedules**
- Upload CSV with `t,q`, or
- Build an easy **ton/day cycle schedule** inside the app and export it.
"""
)

# -------------------------
# plotting helpers
# -------------------------
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
    plt.title("Schedule q(t)")
    plt.plot(t, q)
    if tidx is not None and 0 <= tidx < len(t):
        plt.axvline(t[tidx], linestyle="--")
    plt.xlabel("t (days or step)")
    plt.ylabel("q (ton/day)")
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
    meta = {}
    for key in data.files:
        if key not in ("phi", "k"):
            try:
                meta[key] = data[key]
            except Exception:
                pass
    return data["phi"], data["k"], meta

def load_schedule_csv(uploaded):
    df = pd.read_csv(uploaded)
    df.columns = [c.lower().strip() for c in df.columns]
    if "t" not in df.columns or "q" not in df.columns:
        raise ValueError("CSV must have columns: t, q")
    return df[["t", "q"]].copy()

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.header("1) Inputs")

    input_type = st.radio("phi/k source", ["NPZ (phi,k)", "Eclipse TEXT deck (.DATA + includes)"], index=0)

    if input_type.startswith("NPZ"):
        up_npz = st.file_uploader("Upload NPZ", type=["npz"])
        up_deck = None
        up_includes = []
    else:
        up_deck = st.file_uploader("Upload main deck (.DATA)", type=["data", "DATA"])
        up_includes = st.file_uploader("Upload INCLUDE files (.INC/.GRDECL) (multi)", accept_multiple_files=True)
        up_npz = None

        st.divider()
        st.subheader("Eclipse conversion")
        ecl_mode = st.selectbox("Convert 3D -> 2D", ["layer", "mean"], index=0)
        ecl_layer = st.number_input("Layer index (if layer)", value=0, step=1)
        ecl_kkey = st.text_input("Permeability keyword", value="PERMX")

    st.divider()
    st.header("2) Schedule")

    sched_mode = st.radio("Schedule input", ["Upload CSV (t,q)", "Build cyclic ton/day schedule"], index=1)

    up_csv = None
    schedule_df = None

    if sched_mode.startswith("Upload"):
        up_csv = st.file_uploader("Upload schedule CSV", type=["csv"])
    else:
        st.subheader("Cycle settings (ton/day)")
        total_days = st.number_input("Total duration (days)", value=2000.0)
        dt_days = st.number_input("dt (days per step)", value=10.0)
        cycle_days = st.number_input("Cycle length (days)", value=365.0)

        inj_days = st.number_input("Injection duration in cycle (days)", value=180.0)
        shut_days = st.number_input("Shut-in duration in cycle (days)", value=5.0)
        prod_days = st.number_input("Production duration in cycle (days)", value=180.0)

        Qinj = st.number_input("Injection rate Qinj (ton/day)", value=1000.0)
        Qprod = st.number_input("Production rate Qprod (ton/day)", value=800.0)

        ramp_days = st.number_input("Ramp smoothing (days)", value=20.0)
        ma_window = st.number_input("Optional moving-average window (timesteps)", value=1, step=1)

        build_btn = st.button("Build schedule", type="secondary")

        if build_btn:
            df = build_cycle_schedule_ton_per_day(
                total_days=float(total_days),
                dt_days=float(dt_days),
                cycle_days=float(cycle_days),
                inj_days=float(inj_days),
                shut_days=float(shut_days),
                prod_days=float(prod_days),
                Qinj_tpd=float(Qinj),
                Qprod_tpd=float(Qprod),
                ramp_days=float(ramp_days),
                start_day=0.0,
            )
            if int(ma_window) > 1:
                df["q"] = moving_average(df["q"].to_numpy(np.float32), int(ma_window))
            st.session_state["schedule_df"] = df

        schedule_df = st.session_state.get("schedule_df", None)

        if schedule_df is not None:
            # export buttons
            csv_bytes = schedule_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download schedule CSV", data=csv_bytes, file_name="schedule_cycle_ton_per_day.csv")

            out_xlsx = io.BytesIO()
            with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
                schedule_df.to_excel(w, index=False, sheet_name="schedule")
            st.download_button("Download schedule Excel", data=out_xlsx.getvalue(), file_name="schedule_cycle_ton_per_day.xlsx")

    st.divider()
    st.header("3) Well + params")
    well_mode = st.selectbox("Well placement", ["max_k", "center", "manual"], index=0)
    manual_i = st.number_input("manual i", value=0, step=1)
    manual_j = st.number_input("manual j", value=0, step=1)

    thr_area = st.slider("Area threshold (Sg > thr)", 0.0, 0.5, 0.05, 0.01)

    st.subheader("Model parameters")
    params = dict(DEFAULT_PARAMS)
    show_params = st.checkbox("Edit parameters", value=False)
    if show_params:
        for k in list(params.keys()):
            params[k] = st.number_input(k, value=float(params[k]))

    run_btn = st.button("Run forward prediction", type="primary")

# -------------------------
# Load phi/k
# -------------------------
try:
    if input_type.startswith("NPZ"):
        if up_npz is None:
            st.info("Upload NPZ to begin.")
            st.stop()
        phi, k, meta = load_npz(up_npz)
        src_note = "NPZ"
    else:
        if up_deck is None:
            st.info("Upload .DATA deck to begin.")
            st.stop()
        phi, k, meta = load_eclipse_phi_k_from_uploads(
            up_deck,
            up_includes or [],
            kkey=ecl_kkey,
            mode=ecl_mode,
            layer=int(ecl_layer),
        )
        src_note = f"Eclipse: {meta.get('deck','deck')}"
except Exception as e:
    st.error(f"Failed to read inputs: {e}")
    st.stop()

# -------------------------
# Load schedule
# -------------------------
try:
    if sched_mode.startswith("Upload"):
        if up_csv is None:
            st.info("Upload schedule CSV (t,q), or switch to schedule builder.")
            st.stop()
        schedule_df = load_schedule_csv(up_csv)
    else:
        if schedule_df is None:
            st.info("Click **Build schedule** in the sidebar.")
            st.stop()

    t = schedule_df["t"].to_numpy(dtype=np.float32)
    q = schedule_df["q"].to_numpy(dtype=np.float32)
except Exception as e:
    st.error(f"Failed to read schedule: {e}")
    st.stop()

st.caption(f"Loaded {src_note} | phi/k shape={phi.shape} | schedule steps={len(t)}")

# previews
colA, colB = st.columns(2)
with colA:
    st.subheader("Porosity (phi)")
    st.pyplot(fig_imshow(phi, f"phi | shape={phi.shape}"))
with colB:
    st.subheader("Permeability (k)")
    st.pyplot(fig_imshow(k, f"k | shape={k.shape}"))

st.subheader("Schedule (ton/day)")
st.pyplot(fig_schedule(t, q, tidx=0))

# well selection preview
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

if not run_btn:
    st.stop()

# run model
with st.spinner("Running forward model..."):
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

Nt = len(res.sg_list)
tidx = st.slider("Select timestep", 0, max(0, Nt - 1), min(0, Nt - 1))

left, right = st.columns(2)
with left:
    st.subheader(f"Sg predicted | tidx={tidx}")
    st.pyplot(fig_imshow(res.sg_list[tidx], f"Sg | tidx={tidx}", vmin=0.0, vmax=1.0))
with right:
    st.subheader(f"Pressure surrogate | tidx={tidx}")
    if res.p_list is None:
        st.write("Pressure disabled.")
    else:
        st.pyplot(fig_imshow(res.p_list[tidx], f"p | tidx={tidx}"))

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

# downloads
st.subheader("Download results")
out_npz = io.BytesIO()
sg_stack = np.stack([np.nan_to_num(s, nan=0.0) for s in res.sg_list], axis=0).astype(np.float32)
np.savez_compressed(out_npz, sg=sg_stack, t=res.t, q=res.q, area=res.area, r_eq=res.r_eq)
st.download_button("Download predicted Sg (NPZ)", data=out_npz.getvalue(), file_name="sg_predicted.npz")

out_csv = pd.DataFrame({"t": res.t, "q": res.q, "area": res.area, "r_eq": res.r_eq}).to_csv(index=False).encode("utf-8")
st.download_button("Download time series (CSV)", data=out_csv, file_name="plume_timeseries.csv")
