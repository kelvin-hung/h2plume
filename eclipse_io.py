import io
import re
import zipfile
from pathlib import Path

import numpy as np


def _strip_comments(line: str) -> str:
    # Eclipse sometimes uses -- comments; also support # for robustness
    line = line.split("--", 1)[0]
    line = line.split("#", 1)[0]
    return line.rstrip("\n")


def _parse_repeat_token(tok: str):
    # Eclipse repeat notation: 10*0.25
    m = re.match(r"^(\d+)\*(.+)$", tok)
    if m:
        n = int(m.group(1))
        v = float(m.group(2))
        return [v] * n
    return [float(tok)]


def _parse_numeric_block(lines, start_idx):
    """
    Parse numbers from lines starting at start_idx until encountering a '/'.
    Supports repeat notation.
    Returns (values_array, end_idx_inclusive)
    """
    vals = []
    i = start_idx
    while i < len(lines):
        s = _strip_comments(lines[i]).strip()
        if not s:
            i += 1
            continue

        if "/" in s:
            head = s.split("/", 1)[0].strip()
            if head:
                for tok in head.replace(",", " ").split():
                    vals.extend(_parse_repeat_token(tok))
            return np.array(vals, dtype=np.float32), i

        for tok in s.replace(",", " ").split():
            # ignore non-numeric stray tokens safely
            try:
                vals.extend(_parse_repeat_token(tok))
            except Exception:
                pass

        i += 1

    raise RuntimeError("Numeric block did not terminate with '/'")


def _find_keyword(lines, keyword: str):
    kw = keyword.upper()
    for i, raw in enumerate(lines):
        if _strip_comments(raw).strip().upper() == kw:
            return i
    return None


def _read_specgrid_or_dimens(lines):
    idx = _find_keyword(lines, "SPECGRID")
    if idx is None:
        idx = _find_keyword(lines, "DIMENS")
    if idx is None:
        raise RuntimeError("Could not find SPECGRID or DIMENS.")
    block, _ = _parse_numeric_block(lines, idx + 1)
    if block.size < 3:
        raise RuntimeError("SPECGRID/DIMENS has <3 numbers.")
    nx, ny, nz = int(block[0]), int(block[1]), int(block[2])
    return nx, ny, nz


def _read_keyword_array(lines, keyword: str):
    idx = _find_keyword(lines, keyword)
    if idx is None:
        return None
    arr, _ = _parse_numeric_block(lines, idx + 1)
    return arr


def _read_include_path_from_line(line: str):
    s = _strip_comments(line).strip()
    if not s:
        return None
    # quoted path
    m = re.search(r"'([^']+)'", s)
    if m:
        return m.group(1)
    # otherwise token
    toks = s.replace(",", " ").split()
    if not toks:
        return None
    # allow: FILE.INC /
    if toks[0].upper() == "INCLUDE" and len(toks) > 1:
        return toks[1]
    return toks[0]


def _flatten_deck_lines_from_zip(zip_bytes: bytes, entry_name_hint: str = None, max_depth=50):
    """
    Read a ZIP containing Eclipse deck and INCLUDEs.
    Returns flattened lines for the main .DATA file.
    """
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes), "r")
    names = zf.namelist()

    # choose entry .DATA
    data_candidates = [n for n in names if n.upper().endswith(".DATA")]
    if not data_candidates:
        raise RuntimeError("ZIP does not contain a .DATA file.")
    if entry_name_hint and entry_name_hint in names:
        entry = entry_name_hint
    else:
        # prefer shortest path name
        entry = sorted(data_candidates, key=lambda s: (s.count("/"), len(s)))[0]

    visited = set()

    def read_text(name):
        return zf.read(name).decode("utf-8", errors="ignore").splitlines()

    def resolve_rel(base_name, rel):
        # Eclipse INCLUDE paths often relative to deck folder
        base_dir = "/".join(base_name.split("/")[:-1])
        candidate = f"{base_dir}/{rel}" if base_dir else rel
        # normalize simplistic
        candidate = candidate.replace("//", "/")
        if candidate in names:
            return candidate
        # fallback: search by basename
        b = rel.split("/")[-1]
        hits = [n for n in names if n.split("/")[-1].lower() == b.lower()]
        return hits[0] if hits else None

    def flatten(name, depth=0):
        if depth > max_depth:
            raise RuntimeError("Too deep INCLUDE nesting (possible loop).")
        if name in visited:
            return []
        visited.add(name)

        lines = read_text(name)
        out = []
        i = 0
        while i < len(lines):
            s = _strip_comments(lines[i]).strip()
            if s.upper() == "INCLUDE":
                # look ahead for the include path line
                j = i + 1
                while j < len(lines) and _strip_comments(lines[j]).strip() == "":
                    j += 1

                # Sometimes decks have INCLUDE then "/" then file; handle that:
                if j < len(lines) and _strip_comments(lines[j]).strip() == "/":
                    j += 1
                    while j < len(lines) and _strip_comments(lines[j]).strip() == "":
                        j += 1

                if j >= len(lines):
                    raise RuntimeError(f"INCLUDE without a following path near line {i} in {name}")

                inc_line = _strip_comments(lines[j]).strip()
                inc_rel = _read_include_path_from_line(inc_line)
                if inc_rel is None:
                    raise RuntimeError(f"INCLUDE without path near line {i} in {name}")

                inc_name = resolve_rel(name, inc_rel)
                if inc_name is None:
                    raise RuntimeError(f"INCLUDE file not found in ZIP: {inc_rel} (from {name})")

                out.extend(flatten(inc_name, depth + 1))

                # advance i: skip until we pass the '/' that terminates the include statement
                k = j
                while k < len(lines) and "/" not in lines[k]:
                    k += 1
                i = k + 1
                continue

            out.append(lines[i])
            i += 1
        return out

    return flatten(entry)


def _reshape_eclipse(arr, nx, ny, nz):
    """
    Eclipse grid ordering is typically (K,J,I) in file.
    We reshape -> (nz, ny, nx) then transpose -> (nx, ny, nz) and pick k=0 by default.
    """
    a = arr.reshape((nz, ny, nx))     # (K,J,I)
    a = np.transpose(a, (2, 1, 0))    # (I,J,K)
    return a


def load_eclipse_phi_k_from_uploads(zip_uploaded_file, layer=0, kkey="PERMX", mode="layer"):
    """
    Streamlit uploader -> zip bytes -> flatten deck -> read SPECGRID + PORO + PERMX.
    Returns phi2d, k2d (float32) with inactive masked to NaN if ACTNUM present.
    """
    zip_bytes = zip_uploaded_file.getvalue()
    lines = _flatten_deck_lines_from_zip(zip_bytes)

    nx, ny, nz = _read_specgrid_or_dimens(lines)
    n = nx * ny * nz

    phi = _read_keyword_array(lines, "PORO")
    kx = _read_keyword_array(lines, kkey)
    act = _read_keyword_array(lines, "ACTNUM")  # optional

    if phi is None or kx is None:
        raise RuntimeError(f"Missing PORO or {kkey} in deck.")

    if phi.size != n:
        raise RuntimeError(f"PORO has {phi.size}, expected {n} (nx*ny*nz).")
    if kx.size != n:
        raise RuntimeError(f"{kkey} has {kx.size}, expected {n} (nx*ny*nz).")

    phi3 = _reshape_eclipse(phi.astype(np.float32), nx, ny, nz)
    k3 = _reshape_eclipse(kx.astype(np.float32), nx, ny, nz)

    if mode == "layer":
        kk = int(np.clip(layer, 0, nz - 1))
        phi2 = phi3[:, :, kk].copy()
        k2 = k3[:, :, kk].copy()
    else:
        phi2 = np.nanmean(phi3, axis=2).astype(np.float32)
        k2 = np.nanmean(k3, axis=2).astype(np.float32)

    # ACTNUM masking if present and correct length
    if act is not None and act.size == n:
        act3 = _reshape_eclipse(act.astype(np.float32), nx, ny, nz)
        if mode == "layer":
            act2 = act3[:, :, int(np.clip(layer, 0, nz - 1))]
        else:
            act2 = np.nanmean(act3, axis=2)
        inactive = (act2 <= 0.0) | (~np.isfinite(phi2)) | (~np.isfinite(k2))
        phi2[inactive] = np.nan
        k2[inactive] = np.nan

    return phi2.astype(np.float32), k2.astype(np.float32)
