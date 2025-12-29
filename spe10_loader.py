# spe10_loader.py
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np

# SPE10 Model 2 / SPE102 common dims
SPE10_NX, SPE10_NY, SPE10_NZ = 60, 220, 85
SPE10_N = SPE10_NX * SPE10_NY * SPE10_NZ

@dataclass
class SPE10Props3D:
    nx: int
    ny: int
    nz: int
    phi: np.ndarray   # (nx, ny, nz)
    kx: np.ndarray    # (nx, ny, nz)
    ky: np.ndarray    # (nx, ny, nz)
    kz: np.ndarray    # (nx, ny, nz)

def _read_floats_from_dat(text: str) -> np.ndarray:
    # Fast parsing for whitespace/tab-separated floats
    arr = np.fromstring(text, sep=" ", dtype=np.float64)
    if arr.size == 0:
        # fromstring may return empty if separators are not spaces; fallback:
        arr = np.fromstring(text.replace("\\t", " "), sep=" ", dtype=np.float64)
    return arr

def load_spe10_zip(
    zip_bytes: bytes,
    nx: int = SPE10_NX,
    ny: int = SPE10_NY,
    nz: int = SPE10_NZ,
    phi_name: str = "spe_phi.dat",
    perm_name: str = "spe_perm.dat",
) -> SPE10Props3D:
    """
    Loads SPE10/SPE102-style porosity/permeability files.

    Expected:
      - spe_phi.dat  : N floats (PORO)
      - spe_perm.dat : 3*N floats stacked as [KX, KY, KZ] (mD)

    Returns arrays shaped (nx, ny, nz) in (I, J, K) ordering.
    """
    N = int(nx * ny * nz)

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        if phi_name not in zf.namelist():
            raise FileNotFoundError(f"Missing {phi_name} in zip. Found: {zf.namelist()}")
        if perm_name not in zf.namelist():
            raise FileNotFoundError(f"Missing {perm_name} in zip. Found: {zf.namelist()}")

        phi_txt = zf.read(phi_name).decode("utf-8", errors="ignore")
        perm_txt = zf.read(perm_name).decode("utf-8", errors="ignore")

    phi1 = _read_floats_from_dat(phi_txt)
    perm1 = _read_floats_from_dat(perm_txt)

    if phi1.size != N:
        raise ValueError(f"PORO count mismatch: got {phi1.size}, expected {N} for nx,ny,nz={nx},{ny},{nz}.")
    if perm1.size != 3 * N:
        raise ValueError(f"PERM count mismatch: got {perm1.size}, expected {3*N} (KX,KY,KZ).")

    kx1 = perm1[0:N]
    ky1 = perm1[N:2*N]
    kz1 = perm1[2*N:3*N]

    # The standard SPE10 ordering in the .dat is consistent with Eclipse "K fastest?": in practice,
    # reshaping as (nz, ny, nx) then transposing to (nx, ny, nz) matches common usage.
    def reshape_spe(a1: np.ndarray) -> np.ndarray:
        a3 = a1.reshape((nz, ny, nx)).transpose(2, 1, 0)  # -> (nx, ny, nz)
        return a3.astype(np.float32)

    return SPE10Props3D(
        nx=nx, ny=ny, nz=nz,
        phi=reshape_spe(phi1),
        kx=reshape_spe(kx1),
        ky=reshape_spe(ky1),
        kz=reshape_spe(kz1),
    )

def to_2d(
    arr3: np.ndarray,
    mode: Literal["layer", "mean"] = "layer",
    layer: int = 0,
) -> np.ndarray:
    if arr3.ndim != 3:
        raise ValueError("arr3 must be 3D (nx,ny,nz)")
    nx, ny, nz = arr3.shape
    if mode == "layer":
        k = int(np.clip(layer, 0, nz - 1))
        return arr3[:, :, k].astype(np.float32)
    if mode == "mean":
        return np.nanmean(arr3, axis=2).astype(np.float32)
    raise ValueError("mode must be layer or mean")
