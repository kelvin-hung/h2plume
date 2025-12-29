import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from ve_core import DEFAULT_PARAMS, run_forward, choose_well_ij, prepare_phi_k
from eclipse_io import load_eclipse_phi_k_from_upload

# optional: only for smoothing plots
try:
    from scipy.ndimage import gaussian_filter
except Exception:
    gaussian_filter = None

st.set_page_config(page_title="VE+Darcy+Land Forward Predictor", layout="wide")
st.title("VE + Darcy + Land: Forward plume prediction (paper model)")

st.markdown(
    """
Run the paper forward model **without sg_obs**.

You can load:
- **NPZ** with keys `phi`, `k` (2D arrays)
- **Eclipse deck**: upload a `.DATA` or a `.zip` containing `.DATA` + INCLUDE files

Schedule CSV must have:
- `t` (time or timestep index)
- `q` (signed rate per step; + inject, - produce)
"""
)

# -------------------------
# Helpers
# -------------------------
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
    t = df["t"].to_numpy(dtype=np.float32)
    q = df["q"].to_numpy(dtype=np.float32)
    return t, q

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

def maybe_smooth_for_plot(sg, sigma):
    if sigma <= 0:
        return sg
    if gaussian_filter is None:
        return sg  # scipy not installed
    # smooth only finite region, preserve NaNs outside
    out = np.array(sg, dtype=np.float32, copy=True)
    m = np.isfinite(out)
    if not m.any():
        return out
    filled = np.where(m, out, 0.0)
    w = gaussian_filter(m.astype(np.float32), sigma=sigma)
    s = gaussian_filter(filled, sigma=sigma)
    out = np.where(w > 1e-6, s / np.maximum(w, 1e-6), np.nan)
    return out

# -------------------------
# Sidebar inputs
# -------------------------
with st.sidebar:
    st.header("1) Input type")
    input_type = st.radio("Choose input format", ["NPZ (phi/k)", "Eclipse (.DATA or .zip)"], index=0)

    st.divider()
    st.header("2) Upload inputs")
    if input_type.startswith("NPZ"):
        up_npz = st.file_uploader("phi/k NPZ", type=["npz"])
        up_ecl = None
    else:
        up_ecl = st.file_uploader("Eclipse deck (.DATA) or ZIP", type=["data", "zip", "DATA"])
        up_npz = None

    up_csv = st.file_uploader("schedule CSV", type=["csv"])

    st.divider()
    st.header("3) Eclipse options (if used)")
    ecl_mode = st.selectbox("Convert 3D -> 2D", ["layer", "mean"], index=0)
    ecl_layer = st.number_input("Layer index (if mode=layer)", value=0, step=1)
    ecl_kkey = st.text_input("Permeability keyword", value="PERMX")

    st.divider()
    st.header("4) Well location")
    well_mode = st.selectbox("Well placement", ["max_k", "center", "manual"], index=0)
    manual_i = st.number_input("manual i", value=0, step=1)
    manual_j = st.number_input("manual j", value=0, step=1)

    st.divider()
    st.header("5) Schedule scaling")
    q_scale = st.number_input("Schedule scale factor (multiplies q)", value=1.0)
    normalize_q = st.checkbox("Normalize q to max(|q|)=1", value=False)

    st.divider()
    st.header("6) Output / visualization")
    thr_area = st.slider("Area threshold (Sg > thr)", 0.0, 0.5, 0.05, 0.01)
    smooth_sigma = st.slider("Smooth Sg for plotting (sigma)", 0.0, 3.0, 1.0, 0.25)

    st.divider()
    st.header("7) Paper parameters")
    use_defaults = st.checkbox("Use paper-calibrated defaults", value=True)

    params = dict(DEFAULT_PARAMS)
    if not use_defaults:
        for k in list(params.keys()):
            params[k] = st.number_input(k, value=float(params[k]))

    st.divider()
    run_btn = st.button("Run forward prediction", type="primary")

# -------------------------
# Main logic
# -------------------------
if up_csv is None or (up_npz is None and up_ecl is None):
    st.info("Upload inputs to begin.")
    st.stop()

try:
    # Load phi/k
    if up_npz is not None:
        phi, k, meta = load_npz(up_npz)
        meta_note = "NPZ"
    else:
        phi, k, meta = load_eclipse_phi_k_from_upload(
            up_ecl, kkey=ecl_kkey, mode=ecl_mode, layer=int(ecl_layer)
        )
        meta_note = f"Eclipse: {meta.get('deck_name','deck')}"

    # Load schedule
    t, q = load_schedule_csv(up_csv)

    q = q.astype(np.float32) * np.float32(q_scale)
    if normalize_q:
        m = float(np.max(np.abs(q))) if q.size else 1.0
        if m > 0:
            q = (q / m).astype(np.float32)

except Exception as e:
    st.error(f"Failed to read inputs: {e}")
    st.stop()

st.caption(f"Loaded {meta_note} | phi shape={phi.shape} | k shape={k.shape}")

# quick preview
colA, colB = st.columns(2)
with colA:
    st.subheader("Input: porosity (phi)")
    st.pyplot(fig_imshow(phi, f"phi | shape={phi.shape}"))
with colB:
    st.subheader("Input: permeability (k)")
    st.pyplot(fig_imshow(k, f"k | shape={k.shape}"))

# compute well coordinate for display
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

if not run_btn:
    st.stop()

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
        )
    except Exception as e:
        st.error(f"Run failed: {e}")
        st.stop()

st.success("Done.")

Nt = len(res.sg_list)
tidx = st.slider("Select timestep (tidx)", 0, max(0, Nt - 1), min(0, Nt - 1))

sg_plot = maybe_smooth_for_plot(res.sg_list[tidx], sigma=float(smooth_sigma))

left, right = st.columns(2)
with left:
    st.subheader(f"Sg predicted | tidx={tidx}")
    st.pyplot(fig_imshow(sg_plot, f"Sg predicted | tidx={tidx} (plot-smoothed)", vmin=0.0, vmax=1.0))
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
