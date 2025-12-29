# eclipse_io.py
from __future__ import annotations

import re
import zipfile
import tempfile
from pathlib import Path
import numpy as np

# Keep tempdirs alive during Streamlit run
_TMPDIR_HOLD = []


def strip_comments(line: str) -> str:
    line = line.split("--", 1)[0]
    line = line.split("#", 1)[0]
    return line.strip()


def parse_repeat_token(tok: str) -> list[float]:
    """
    Eclipse repeat notation:
      10*0.25 -> [0.25]*10
    Robust: ignore non-numeric tokens (e.g., F, T)
    """
    tok = tok.strip()
    if not tok:
        return []
    m = re.match(r"^(\d+)\*(.+)$", tok)
    if m:
        n = int(m.group(1))
        try:
            v = float(m.group(2))
        except Exception:
            return []
        return [v] * n
    try:
        return [float(tok)]
    except Exception:
        return []


def parse_numeric_block(lines: list[str], start_idx: int) -> tuple[np.ndarray, int]:
    """
    Parse numbers from lines until encountering '/'.
    Ignores non-numeric tokens safely.
    """
    vals: list[float] = []
    i = start_idx
    while i < len(lines):
        s = strip_comments(lines[i])
        if not s:
            i += 1
            continue

        if "/" in s:
            head = s.split("/", 1)[0].strip()
            if head:
                for tok in head.replace(",", " ").split():
                    vals.extend(parse_repeat_token(tok))
            return np.array(vals, dtype=np.float32), i

        for tok in s.replace(",", " ").split():
            vals.extend(parse_repeat_token(tok))
        i += 1

    raise RuntimeError("Numeric block did not terminate with '/'")


def read_include_path(line: str) -> str | None:
    s = strip_comments(line)
    m = re.search(r"'([^']+)'", s)
    if m:
        return m.group(1)
    toks = s.split()
    return toks[1] if len(toks) > 1 else None


def flatten_deck_with_includes(entry_path: Path, max_depth: int = 30) -> list[str]:
    visited = set()

    def _flatten(p: Path, depth: int) -> list[str]:
        if depth > max_depth:
            raise RuntimeError("Too deep INCLUDE nesting (possible loop).")

        p = p.resolve()
        key = str(p)
        if key in visited:
            return []
        visited.add(key)

        txt = p.read_text(errors="ignore")
        lines = txt.splitlines()

        out: list[str] = []
        i = 0
        while i < len(lines):
            s_upper = strip_comments(lines[i]).upper()

            if s_upper == "INCLUDE":
                j = i + 1
                while j < len(lines) and not strip_comments(lines[j]):
                    j += 1
                if j >= len(lines):
                    raise RuntimeError(f"INCLUDE without path near line {i} in {p.name}")

                inc_rel = read_include_path(lines[j])
                if inc_rel is None:
                    raise RuntimeError(f"INCLUDE without path near line {j} in {p.name}")

                inc_path = (p.parent / inc_rel).resolve()
                out.extend(_flatten(inc_path, depth + 1))

                # advance until we pass the line that contains '/'
                k = j
                while k < len(lines) and "/" not in lines[k]:
                    k += 1
                i = k + 1
                continue

            out.append(lines[i])
            i += 1

        return out

    return _flatten(entry_path, 0)


def find_keyword(lines: list[str], keyword: str) -> int | None:
    kw = keyword.upper()
    for i, raw in enumerate(lines):
        if strip_comments(raw).upper() == kw:
            return i
    return None


def read_keyword_array(lines: list[str], keyword: str) -> np.ndarray | None:
    idx = find_keyword(lines, keyword)
    if idx is None:
        return None
    arr, _ = parse_numeric_block(lines, idx + 1)
    return arr


def read_specgrid(lines: list[str]) -> tuple[int, int, int]:
    """
    SPECGRID may contain flags like:
      46 112 22 1 F /
    We only take the first 3 numeric values.
    """
    idx = find_keyword(lines, "SPECGRID")
    if idx is None:
        idx = find_keyword(lines, "DIMENS")
    if idx is None:
        raise RuntimeError("Could not find SPECGRID or DIMENS.")

    block, _ = parse_numeric_block(lines, idx + 1)
    if block.size < 3:
        raise RuntimeError("SPECGRID/DIMENS has <3 numeric values.")
    nx, ny, nz = int(block[0]), int(block[1]), int(block[2])
    return nx, ny, nz


def to_2d(arr3d: np.ndarray, nx: int, ny: int, nz: int, mode: str, layer: int) -> np.ndarray:
    a = arr3d.reshape((nz, ny, nx))        # (K, J, I)
    a = np.transpose(a, (2, 1, 0))         # -> (I, J, K)
    if mode == "layer":
        k = int(np.clip(layer, 0, nz - 1))
        return a[:, :, k]
    if mode == "mean":
        return np.nanmean(a, axis=2)
    raise ValueError("mode must be 'layer' or 'mean'")


def _prepare_workspace_from_upload(uploaded) -> Path:
    """
    Returns deck_path (Path).
    Supports .DATA or .zip (deck + INCLUDEs).
    """
    name = (uploaded.name or "").lower()

    td = tempfile.TemporaryDirectory()
    _TMPDIR_HOLD.append(td)  # keep alive
    root = Path(td.name)

    if name.endswith(".zip"):
        zpath = root / "input.zip"
        zpath.write_bytes(uploaded.getvalue())
        with zipfile.ZipFile(zpath, "r") as zf:
            zf.extractall(root)

        data_files = sorted(root.rglob("*.DATA")) + sorted(root.rglob("*.data"))
        if not data_files:
            raise RuntimeError("ZIP does not contain any .DATA file.")
        return data_files[0]

    # assume it's a deck file
    deck_path = root / (uploaded.name if uploaded.name else "DECK.DATA")
    deck_path.write_bytes(uploaded.getvalue())
    return deck_path


def load_eclipse_phi_k_from_upload(
    uploaded,
    *,
    kkey: str = "PERMX",
    mode: str = "layer",
    layer: int = 0
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Streamlit-friendly loader.
    Output: phi2d, k2d, meta
    """
    deck = _prepare_workspace_from_upload(uploaded)
    lines = flatten_deck_with_includes(deck)

    nx, ny, nz = read_specgrid(lines)
    n = nx * ny * nz

    phi = read_keyword_array(lines, "PORO")
    kx = read_keyword_array(lines, kkey.upper())
    act = read_keyword_array(lines, "ACTNUM")

    if phi is None:
        raise RuntimeError("PORO not found in deck (after INCLUDE flatten).")
    if kx is None:
        raise RuntimeError(f"{kkey.upper()} not found in deck (after INCLUDE flatten).")

    if phi.size != n:
        raise RuntimeError(f"PORO has {phi.size} values, expected {n} (nx*ny*nz).")
    if kx.size != n:
        raise RuntimeError(f"{kkey.upper()} has {kx.size} values, expected {n} (nx*ny*nz).")

    # Apply ACTNUM mask if present
    if act is not None and act.size == n:
        mask = (act.reshape((n,)) > 0)
        phi = np.where(mask, phi, np.nan).astype(np.float32)
        kx = np.where(mask, kx, np.nan).astype(np.float32)

    phi2 = to_2d(phi, nx, ny, nz, mode=mode, layer=int(layer)).astype(np.float32)
    k2 = to_2d(kx, nx, ny, nz, mode=mode, layer=int(layer)).astype(np.float32)

    meta = {
        "deck_name": deck.name,
        "nx": nx, "ny": ny, "nz": nz,
        "mode": mode, "layer": int(layer),
        "kkey": kkey.upper(),
    }
    return phi2, k2, meta
