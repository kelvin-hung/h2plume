# eclipse_io.py
from __future__ import annotations

import re
import tempfile
from pathlib import Path
import numpy as np

_TMPDIR_HOLD = []


def strip_comments(line: str) -> str:
    line = line.split("--", 1)[0]
    line = line.split("#", 1)[0]
    return line.strip()


def parse_repeat_token(tok: str) -> list[float]:
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
    idx = find_keyword(lines, "SPECGRID")
    if idx is None:
        idx = find_keyword(lines, "DIMENS")
    if idx is None:
        raise RuntimeError("Could not find SPECGRID or DIMENS.")
    block, _ = parse_numeric_block(lines, idx + 1)
    if block.size < 3:
        raise RuntimeError("SPECGRID/DIMENS has <3 numeric values.")
    return int(block[0]), int(block[1]), int(block[2])


def to_2d(arr3d: np.ndarray, nx: int, ny: int, nz: int, mode: str, layer: int) -> np.ndarray:
    a = arr3d.reshape((nz, ny, nx))      # (K, J, I)
    a = np.transpose(a, (2, 1, 0))       # -> (I, J, K)
    if mode == "layer":
        k = int(np.clip(layer, 0, nz - 1))
        return a[:, :, k]
    if mode == "mean":
        return np.nanmean(a, axis=2)
    raise ValueError("mode must be 'layer' or 'mean'")


def _write_upload(work: Path, uploaded) -> Path:
    # Preserve subfolders if present in name
    name = uploaded.name or "uploaded.inc"
    p = work / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(uploaded.getvalue())
    return p


def _build_file_index(work: Path) -> dict[str, Path]:
    """
    Map lowercase basename -> path, and also lowercase full relative path -> path.
    Helps resolve INCLUDE 'folder/file.INC' even if uploaded flat.
    """
    idx: dict[str, Path] = {}
    for p in work.rglob("*"):
        if p.is_file():
            rel = str(p.relative_to(work)).replace("\\", "/").lower()
            idx[rel] = p
            idx[p.name.lower()] = p
    return idx


def _extract_include_target(lines: list[str], i: int) -> tuple[str, int]:
    """
    Handles patterns:
      INCLUDE
      'file.INC' /
    or
      INCLUDE 'file.INC' /
    Returns (path_string, end_index_after_include_block)
    """
    # Case 1: path on same line as INCLUDE
    s = strip_comments(lines[i])
    m = re.search(r"'([^']+)'", s)
    if m:
        inc = m.group(1)
        # advance until line containing '/'
        k = i
        while k < len(lines) and "/" not in lines[k]:
            k += 1
        return inc, k + 1

    # Case 2: path on subsequent non-empty line(s)
    j = i + 1
    while j < len(lines) and not strip_comments(lines[j]):
        j += 1
    if j >= len(lines):
        raise RuntimeError(f"INCLUDE without path near line {i}")

    s2 = strip_comments(lines[j])
    m2 = re.search(r"'([^']+)'", s2)
    if m2:
        inc = m2.group(1)
    else:
        toks = s2.split()
        if not toks:
            raise RuntimeError(f"INCLUDE without path near line {i}")
        inc = toks[0]

    # skip until '/'
    k = j
    while k < len(lines) and "/" not in lines[k]:
        k += 1
    return inc, k + 1


def flatten_deck_with_includes(deck_path: Path, work_root: Path, file_index: dict[str, Path], max_depth: int = 40) -> list[str]:
    visited = set()

    def _resolve_include(cur_file: Path, inc: str) -> Path:
        # try relative to current file
        p1 = (cur_file.parent / inc).resolve()
        if p1.exists():
            return p1
        # try by relative path key in index
        key_rel = inc.replace("\\", "/").lstrip("/").lower()
        if key_rel in file_index:
            return file_index[key_rel]
        # try basename
        base = Path(inc).name.lower()
        if base in file_index:
            return file_index[base]
        raise RuntimeError(f"Missing INCLUDE file: {inc}")

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
            if s_upper.startswith("INCLUDE"):
                inc, next_i = _extract_include_target(lines, i)
                inc_path = _resolve_include(p, inc)
                out.extend(_flatten(inc_path, depth + 1))
                i = next_i
                continue
            out.append(lines[i])
            i += 1
        return out

    return _flatten(deck_path, 0)


def load_eclipse_phi_k_from_uploads(
    deck_upload,
    include_uploads: list,
    *,
    kkey: str = "PERMX",
    mode: str = "layer",
    layer: int = 0
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Streamlit-friendly loader for TEXT decks.
    - deck_upload: .DATA or .GRDECL etc (main deck)
    - include_uploads: list of extra files user uploads (INC/GRDECL)
    """
    td = tempfile.TemporaryDirectory()
    _TMPDIR_HOLD.append(td)
    work = Path(td.name)

    deck_path = _write_upload(work, deck_upload)
    for f in include_uploads or []:
        _write_upload(work, f)

    file_index = _build_file_index(work)

    lines = flatten_deck_with_includes(deck_path, work, file_index)
    nx, ny, nz = read_specgrid(lines)
    n = nx * ny * nz

    phi = read_keyword_array(lines, "PORO")
    kx = read_keyword_array(lines, kkey.upper())
    act = read_keyword_array(lines, "ACTNUM")

    if phi is None:
        raise RuntimeError("PORO not found. This deck may define properties via EQUALS/BOX/COPY; export explicit PORO first.")
    if kx is None:
        raise RuntimeError(f"{kkey.upper()} not found. Try PERMX/PERMX or export explicit permeability array.")

    if phi.size != n:
        raise RuntimeError(f"PORO has {phi.size} values, expected {n} (nx*ny*nz).")
    if kx.size != n:
        raise RuntimeError(f"{kkey.upper()} has {kx.size} values, expected {n} (nx*ny*nz).")

    if act is not None and act.size == n:
        mask = (act.reshape((n,)) > 0)
        phi = np.where(mask, phi, np.nan).astype(np.float32)
        kx = np.where(mask, kx, np.nan).astype(np.float32)

    phi2 = to_2d(phi, nx, ny, nz, mode=mode, layer=int(layer)).astype(np.float32)
    k2 = to_2d(kx, nx, ny, nz, mode=mode, layer=int(layer)).astype(np.float32)

    meta = {"nx": nx, "ny": ny, "nz": nz, "mode": mode, "layer": int(layer), "kkey": kkey.upper(), "deck": deck_path.name}
    return phi2, k2, meta
