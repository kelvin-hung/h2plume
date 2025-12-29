# schedule_tools.py
from __future__ import annotations
import numpy as np
import pandas as pd


def build_cycle_schedule_ton_per_day(
    total_days: float,
    dt_days: float,
    cycle_days: float,
    inj_days: float,
    shut_days: float,
    prod_days: float,
    Qinj_tpd: float,
    Qprod_tpd: float = 0.0,
    ramp_days: float = 0.0,
    start_day: float = 0.0,
) -> pd.DataFrame:
    """
    Build repeating cycle schedule in ton/day.
    Pattern per cycle:
      inj ( +Qinj ) for inj_days
      shut ( 0 )   for shut_days
      prod ( -Qprod ) for prod_days
      idle ( 0 ) for remaining days of cycle

    ramp_days: linear ramp applied at step changes to avoid sharp discontinuities.
    """
    if dt_days <= 0:
        raise ValueError("dt_days must be > 0")
    if total_days <= start_day:
        raise ValueError("total_days must be > start_day")
    if cycle_days <= 0:
        raise ValueError("cycle_days must be > 0")
    if inj_days < 0 or shut_days < 0 or prod_days < 0:
        raise ValueError("durations must be >= 0")
    if inj_days + shut_days + prod_days > cycle_days + 1e-9:
        raise ValueError("inj_days + shut_days + prod_days cannot exceed cycle_days")

    # Discrete simulation time grid
    Nt = int(np.floor((total_days - start_day) / dt_days)) + 1
    t = start_day + np.arange(Nt, dtype=np.float32) * np.float32(dt_days)

    # Daily schedule used for clean cycling + ramping
    days = np.arange(int(np.ceil(total_days - start_day)) + 1, dtype=np.float32) + np.float32(start_day)
    qd = np.zeros_like(days, dtype=np.float32)

    for i, day in enumerate(days):
        pos = float((day - start_day) % cycle_days)
        if pos < inj_days:
            qd[i] = float(Qinj_tpd)
        elif pos < inj_days + shut_days:
            qd[i] = 0.0
        elif pos < inj_days + shut_days + prod_days:
            qd[i] = -float(Qprod_tpd)
        else:
            qd[i] = 0.0

    # Apply linear ramp at change points
    rlen = int(np.round(max(0.0, ramp_days)))
    if rlen > 0:
        q2 = qd.copy()
        change_idx = np.where(np.abs(np.diff(qd)) > 1e-12)[0]
        for c in change_idx:
            i0 = c + 1
            i1 = min(i0 + rlen, len(q2))
            q0 = qd[c]
            q1 = qd[c + 1]
            n = i1 - i0
            if n > 0:
                q2[i0:i1] = np.linspace(q0, q1, n, endpoint=False, dtype=np.float32)
        qd = q2

    # Sample daily schedule at timestep points
    idx = np.clip(np.round(t - start_day).astype(int), 0, len(qd) - 1)
    q = qd[idx].astype(np.float32)

    return pd.DataFrame({"t": t.astype(np.float32), "q": q.astype(np.float32)})


def moving_average(q: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return q
    window = int(window)
    kernel = np.ones(window, dtype=np.float32) / np.float32(window)
    # pad edges
    pad = window // 2
    qp = np.pad(q, (pad, pad), mode="edge")
    return np.convolve(qp, kernel, mode="valid").astype(np.float32)
