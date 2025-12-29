# eclipse_loader.py
from __future__ import annotations

import io
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

@dataclass
class EclipseCase:
    root: str
    egrid: Path
    init: Optional[Path]

def _find_cases(folder: Path) -> List[EclipseCase]:
    files = list(folder.rglob("*"))
    egrids = {}
    inits = {}
    for f in files:
        if not f.is_file():
            continue
        suf = f.suffix.upper()
        if suf == ".EGRID":
            egrids[f.stem] = f
        elif suf == ".INIT":
            inits[f.stem] = f

    cases = []
    for root, egrid in egrids.items():
        cases.append(EclipseCase(root=root, egrid=egrid, init=inits.get(root)))
    cases.sort(key=lambda c: c.root.lower())
    return cases

def list_case_roots(folder: Path) -> List[str]:
    return [c.root for c in _find_cases(folder)]

def extract_zip_to_temp(zip_bytes: bytes) -> Path:
    td = Path(tempfile.mkdtemp(prefix="eclzip_"))
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        zf.extractall(td)
    return td

def load_phi_k_from_eclipse(
    folder: Path,
    root: str,
    kkey: str = "PERMX",
    layer: int = 0,
    mode: str = "layer",  # layer|mean
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Requires: pip install resdata
    Reads PORO + PERMX (or chosen kkey) from INIT and ACTNUM from EGRID if available.
    Returns (phi2d, k2d, mask2d).
    """
    try:
        from resdata.ecl.eclfile import EclFile  # type: ignore
        from resdata.ecl.grid import EclGrid     # type: ignore
    except Exception as e:
        raise RuntimeError(f"Missing dependency 'resdata'. Install it. Import error: {e}")

    case = next((c for c in _find_cases(folder) if c.root == root), None)
    if case is None:
        raise ValueError(f"Could not find case root='{root}' in extracted zip folder.")
    if case.init is None:
        raise ValueError(f"Found {case.egrid.name} but missing {root}.INIT (needed for PORO/PERM).")

    grid = EclGrid(str(case.egrid))
    nx, ny, nz = grid.getNX(), grid.getNY(), grid.getNZ()
    n = nx * ny * nz

    init = EclFile(str(case.init))

    def read_kw(name: str) -> Optional[np.ndarray]:
        name = name.upper()
        if not init.has_kw(name):
            return None
        kw = init[name][0]
        return np.asarray(kw, dtype=np.float32).copy()

    poro = read_kw("PORO")
    perm = read_kw(kkey)
    if poro is None:
        raise ValueError("INIT does not contain PORO")
    if perm is None:
        raise ValueError(f"INIT does not contain {kkey}")

    if poro.size != n or perm.size != n:
        raise ValueError(f"Keyword sizes mismatch. Expected {n}, got PORO={poro.size}, {kkey}={perm.size}")

    # Convert to (nx, ny, nz)
    poro3 = poro.reshape((nz, ny, nx)).transpose(2, 1, 0)
    perm3 = perm.reshape((nz, ny, nx)).transpose(2, 1, 0)

    try:
        actnum = np.asarray(grid.actnum, dtype=np.int32).reshape((nz, ny, nx)).transpose(2, 1, 0) > 0
    except Exception:
        actnum = np.isfinite(poro3) & np.isfinite(perm3) & (poro3 > 0) & (perm3 > 0)

    if mode == "layer":
        k = int(np.clip(layer, 0, nz - 1))
        phi2 = np.where(actnum[:, :, k], poro3[:, :, k], np.nan).astype(np.float32)
        k2 = np.where(actnum[:, :, k], perm3[:, :, k], np.nan).astype(np.float32)
        mask2 = np.isfinite(phi2) & np.isfinite(k2)
        return phi2, k2, mask2
    if mode == "mean":
        phi2 = np.nanmean(np.where(actnum, poro3, np.nan), axis=2).astype(np.float32)
        k2 = np.nanmean(np.where(actnum, perm3, np.nan), axis=2).astype(np.float32)
        mask2 = np.isfinite(phi2) & np.isfinite(k2)
        return phi2, k2, mask2

    raise ValueError("mode must be 'layer' or 'mean'")
