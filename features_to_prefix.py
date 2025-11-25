from typing import List, Dict
from pathlib import Path
import pandas as pd
import numpy as np
import math

def read_csv_strict(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def get_season(df):
    season_col = None
    for c in df.columns:
        if c.strip().lower() == "season":
            season_col = c
            break
    if season_col is None or len(df) == 0:
        return None
    
    val = str(df.iloc[0][season_col]).strip().lower()
    if not val:
        return None
    return val if val in {"spring", "summer", "autumn", "winter"} else None

def build_prefix_tokens(feat: Dict[str, object]) -> List[str]:
    tokens = [
        f"KEY_{feat['key_idx']}",
        "MODE_MAJ" if feat["mode_major"] else "MODE_MIN",
        f"BPM_{feat['bpm']}",
        feat["reg"],
        f"RHY_{feat['rhy_idx']}",
        f"DENS_{feat['dens_idx']}",
        f"CHR_{feat['chr_idx']}",
        "BAR",
        f"POS_{feat['pos_idx']}",
    ]

    order = ["KEY","MODE","BPM","REG","RHY","DENS","CHR","BAR","POS"]
    ordered = []
    for o in order:
        for t in tokens:
            if t.startswith(o):
                ordered.append(t)
                break
    return ordered

def session_to_features(df: pd.DataFrame, season: str) -> Dict[str, object]:
    df_sorted = df.sort_values("StrokeIndex")

    first = df_sorted.iloc[0]
    last = df_sorted.iloc[-1]

    start_x = float(first["Start_X"])
    start_y = float(first["Start_Y"])
    start_z = float(first["Start_Z"])

    end_x = float(last["End_X"])
    end_y = float(last["End_Y"])
    end_z = float(last["End_Z"])

    idx_max = df["Count"].astype(float).idxmax()
    dom = df.loc[idx_max]

    r_i = float(dom["ColorR"])
    g_i = float(dom["ColorG"])
    b_i = float(dom["ColorB"])

    mask = (
        (df["ColorR"] == r_i) &
        (df["ColorG"] == g_i) &
        (df["ColorB"] == b_i)
    )
    a_i = float(df.loc[mask, "ColorA"].astype(float).median())

    alpha_series = None
    brush_series = None

    if "ColorA" in df_sorted.columns:
        alpha_series = df_sorted["ColorA"].astype(float)

    if "BrushSize" in df_sorted.columns:
        brush_series = df_sorted["BrushSize"].astype(float)

    # KEY
    _KEYS = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    _KEY2IDX = {k:i for i,k in enumerate(_KEYS)}
    BASE = {
        (1.0, 0.0, 0.0): "C", # red
        (1.0, 0.9216, 0.0157): "D", # yellow
        (0.0, 1.0, 0.0): "A", # green
        (0.0, 1.0, 1.0): "E", # cyan
        (0.0, 0.0, 1.0): "B", # blue
    } # Scriabin Key
    eps = 1e-4
    base = None
    for (tr, tg, tb), k in BASE.items():
        # picked rgb - target rgb
        if abs(r_i - tr) <= eps and abs(g_i - tg) <= eps and abs(b_i - tb) <= eps:
            base = k
            break
    if base is None:
        base = "C"

    offset = int(round(a_i * 4)) - 2 # -2 ~ +2
    idx = (_KEY2IDX[base] + offset) % 12
    key_name = _KEYS[idx]
    key = idx
    print("[DEBUG] base:", base, "alpha:", a_i, "offset:", offset, "key_name:", key_name)

    # MODE
    if season in {"spring", "summer"}:
        mode_major = True
    else:
        mode_major = False

    # BPM
    luma = (0.299 * (r_i**2) + 0.587 * (g_i**2) + 0.114 * (b_i**2)) # 0~1
    raw_bpm = 60 + luma * (140 - 60) # 60~140
    allowed = [60, 80, 100, 120, 140]
    bpm = min(allowed, key=lambda t: abs(t - raw_bpm))

    # REG
    dx = end_x - start_x
    dy = end_y - start_y
    dz = end_z - start_z
    D = math.sqrt(dx*dx + dy*dy + dz*dz) # distance(3D)
    norm_D = D / (1.0 + D)
    if norm_D < (1.0/3.0):
        reg = "REG_LOW"
    elif norm_D < (2.0/3.0):
        reg = "REG_MID"
    else:
        reg = "REG_HIGH"

    # RHY
    undo_total = float(last.get("TotalUndoCount", 0.0))
    stroke_count = len(df_sorted)
    ratio = undo_total / max(1.0, stroke_count) # 전체 중 undo 비율
    rhy = int(round(2.0 * min(1.0, ratio)))
    rhy = max(0, min(2, rhy))

    # DENS
    BRUSH_MIN  = 0.001
    BRUSH_MAX = 0.05
    dens = 1
    if "BrushSize" in df_sorted.columns:
        if len(brush_series):
            bs_min = float(brush_series.min())
            bs_max = float(brush_series.max())
            if bs_min == bs_max:
                bs = bs_min
            else:
                bins = np.linspace(BRUSH_MIN, BRUSH_MAX, 11) # edges
                hist, edges = np.histogram(brush_series, bins=bins)
                k = int(hist.argmax())
                bs = float((edges[k] + edges[k+1]) / 2.0)
                
            bs = max(BRUSH_MIN, min(BRUSH_MAX, bs))
            norm_bs = (bs - BRUSH_MIN) / (BRUSH_MAX - BRUSH_MIN)
            norm_bs = max(0.0, min(1.0, norm_bs))
            
            if norm_bs < (1.0/3.0):
                dens = 0
            elif norm_bs < (2.0/3.0):
                dens = 1
            else:
                dens = 2
        else:
            dens = 1

    # CHR
    chro = 1
    if (brush_series is not None) and (alpha_series is not None) and len(brush_series) and len(alpha_series):
        idx_max = brush_series.idxmax()
        a = float(alpha_series.loc[idx_max])
        a = max(0.0, min(1.0, a))
        if a < (1.0/3.0):
            chro = 0
        elif a < (2.0/3.0):
            chro = 1
        else:
            chro = 2

    pos = 0

    return {
        "key_idx": key,
        "bpm": bpm,
        "mode_major": mode_major,
        "reg": reg,
        "rhy_idx": rhy,
        "dens_idx": dens,
        "chr_idx": chro,
        "pos_idx": pos,
    }
