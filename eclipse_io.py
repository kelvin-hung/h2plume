import re
import numpy as np
import streamlit as st
from pathlib import Path
import tempfile
import zipfile


def strip_comments(line: str) -> str:
    line = line.split("--", 1)[0]
    line = line.split("#", 1)[0]
    return line.strip()


def _try_float(tok: str):
    # ignore common non-numeric tokens in SPECGRID lines (e.g., 'F', 'T')
    t = tok.strip().strip(",")
    if not t:
        return None
    if t.upper() in {"F", "T"}:
        return None
    # Eclipse repeat: 10*0.25
    m = re.match(r"^(\d+)\*(.+)$", t)
    if m:
        n = int(m.group(1))
        v = float(m.group(2))
        return [v] * n
    # normal float
    try:
        return [float(t)]
    except Exception:
        return None


def parse_numeric_block(lines, start_idx, strict=False):
    vals = []
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
                    out = _try_float(tok)
                    if out is None:
                        if strict:
                            raise ValueError(f"Non-numeric token '{tok}' in numeric block.")
                        continue
                    vals.extend(out)
            return np.array(vals, dtype=np.float32), i

        for tok in s.replace(",", " ").split():
            out = _try_float(tok)
            if out is None:
                if strict:
                    raise ValueError(f"Non-numeric token '{tok}' in numeric block.")
                continue
            vals.extend(out)
        i += 1

    raise RuntimeError("Block did not terminate with '/'")


def read_include_path(line: str):
    s = strip_comments(line)
    m = re.search(r"'([^']+)'", s)
    if m:
        return m.group(1)
    toks = s.split()
    return toks[1] if len(toks) > 1 else None


def flatten_deck_with_includes(entry_path: Path, max_depth: int = 30) -> list[str]:
    visited = set()

    def _flatten(p: Path, depth: int):
        if depth > max_depth:
            raise RuntimeError("Too deep INCLUDE nesting (possible loop).")
        p = p.resolve()
        key = str(p)
        if key in visited:
            return []
        visited.add(key)

        txt = p.read_text(errors="ignore")
        lines = txt.splitlines()
        out = []
        i = 0
        while i < len(lines):
            s = strip_comments(lines[i]).upper()
            if s == "INCLUDE":
                j = i + 1
                while j < len(lines) and not strip_comments(lines[j]):
                    j += 1
                inc_rel = read_include_path(lines[j]) if j < len(lines) else None
                if inc_rel is None:
                    raise RuntimeError(f"INCLUDE without path near line {i} in {p.name}")
                inc_path = (p.parent / inc_rel).resolve()
                out.extend(_flatten(inc_path, depth + 1))

                # skip include statement until '/'
                k = j
                while k < len(lines) and "/" not in lines[k]:
                    k += 1
                i = k + 1
                continue
            else:
                out.append(lines[i])
                i += 1
        return out

    return _flatten(entry_path, 0)


def find_keyword(lines, keyword: str):
    kw = keyword.upper()
    for i, raw in enumerate(lines):
        if strip_comments(raw).upper() == kw:
            return i
    return None


def read_specgrid(lines):
    idx = find_keyword(lines, "SPECGRID")
    if idx is None:
        idx = find_keyword(lines, "DIMENS")
    if idx is None:
        raise RuntimeError("Could not find SPECGRID or DIMENS.")

    block, _ = parse_numeric_block(lines, idx + 1, strict=False)
    if block.size < 3:
        raise RuntimeError("SPECGRID/DIMENS has <3 numeric values.")
    nx, ny, nz = int(block[0]), int(block[1]), int(block[2])
    return nx, ny, nz


def read_keyword_array(lines, keyword: str, strict=True):
    idx = find_keyword(lines, keyword)
    if idx is None:
        return None
    arr, _ = parse_numeric_block(lines, idx + 1, strict=strict)
    return arr


def to_2d(arr3d, nx, ny, nz, mode: str, layer: int):
    a = arr3d.reshape((nz, ny, nx))  # (K,J,I)
    a = np.transpose(a, (2, 1, 0))   # -> (I,J,K)
    if mode == "layer":
        k = int(np.clip(layer, 0, nz - 1))
        return a[:, :, k]
    if mode == "mean":
        return np.nanmean(a, axis=2)
    raise ValueError("mode must be 'layer' or 'mean'")


def load_eclipse_phi_k(deck_path: Path, mode="layer", layer=0, kkey="PERMX"):
    lines = flatten_deck_with_includes(deck_path)
    nx, ny, nz = read_specgrid(lines)
    n = nx * ny * nz

    phi = read_keyword_array(lines, "PORO", strict=True)
    kx = read_keyword_array(lines, kkey, strict=True)
    act = read_keyword_array(lines, "ACTNUM", strict=False)

    if phi is None or kx is None:
        raise RuntimeError(f"Missing PORO or {kkey} in deck/INCLUDE files.")

    if phi.size != n:
        raise RuntimeError(f"PORO has {phi.size} values, expected {n}.")
    if kx.size != n:
        raise RuntimeError(f"{kkey} has {kx.size} values, expected {n}.")

    if act is not None and act.size == n:
        m = act.reshape((n,)) > 0
        phi = np.where(m, phi, np.nan).astype(np.float32)
        kx = np.where(m, kx, np.nan).astype(np.float32)

    phi2 = to_2d(phi, nx, ny, nz, mode, layer).astype(np.float32)
    k2 = to_2d(kx, nx, ny, nz, mode, layer).astype(np.float32)
    return phi2, k2


def load_eclipse_phi_k_from_upload(uploaded_file, mode="layer", layer=0, kkey="PERMX"):
    # Supports .DATA or .zip
    name = uploaded_file.name
    suffix = Path(name).suffix.lower()

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        if suffix == ".zip":
            zpath = td / "deck.zip"
            zpath.write_bytes(uploaded_file.getvalue())
            with zipfile.ZipFile(zpath, "r") as z:
                z.extractall(td)
            # find a .DATA file
            data_files = list(td.rglob("*.DATA")) + list(td.rglob("*.data"))
            if not data_files:
                raise RuntimeError("ZIP contains no .DATA file.")
            deck = data_files[0]
        else:
            # assume it's a deck
            deck = td / Path(name).name
            deck.write_bytes(uploaded_file.getvalue())

        return load_eclipse_phi_k(deck, mode=mode, layer=layer, kkey=kkey)
