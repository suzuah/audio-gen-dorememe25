from typing import List, Dict
from pathlib import Path
import pandas as pd
import math

def read_csv_strict(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

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

_KEYS = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
_KEY2IDX = {k:i for i,k in enumerate(_KEYS)}

def _classify_color_5(r: int, g: int, b: int) -> str:
    mx = max(r, g, b); mn = min(r, g, b)
    if mx > 220 and (mx - mn) < 15: return "white"
    if r >= g and r >= b:
        if r > 180 and g > 180 and b < 120: return "yellow"
        return "red"
    if g >= r and g >= b: return "green"
    return "blue"

_BPM_MAP = {"red": 120, "yellow": 100, "green": 100, "blue": 80, "white": 60}
_COLOR_KEY = {"red": "C", "yellow":"D#", "green":"F", "blue":"G#", "white":"A#"}

def session_to_prefix(df: pd.DataFrame) -> Dict[str, object]:
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

    r_val = float(dom["ColorR"])
    g_val = float(dom["ColorG"])
    b_val = float(dom["ColorB"])

    if max(r_val, g_val, b_val) <= 1.0:
        r_i = int(round(r_val * 255))
        g_i = int(round(g_val * 255))
        b_i = int(round(b_val * 255))
    else:
        r_i = int(round(r_val))
        g_i = int(round(g_val))
        b_i = int(round(b_val))

    warm = (r_i >= g_i and r_i >= b_i)
    cool = (b_i >= r_i and b_i >= g_i)
    mode_major = not (cool and not warm)

    """
    instability = float(undo_total)
    p_minor = 1.0 / (1.0 + math.exp(-instability))
    mode_major = (p_minor < 0.5)
    """

    dx = end_x - start_x
    dy = end_y - start_y
    dz = end_z - start_z
    D = (dx*dx + dy*dy + dz*dz) ** 0.5

    norm_D = D / (1.0 + D)
    if norm_D < (1.0/3.0):
        reg = "REG_LOW"
    elif norm_D < (2.0/3.0):
        reg = "REG_MID"
    else:
        reg = "REG_HIGH"

    stroke_count = len(df_sorted)
    complexity_score = stroke_count * D
    score = math.log1p(complexity_score)
    edge_density = score / (1.0 + score)

    dens = int(round(edge_density * 2))
    if dens < 0: dens = 0
    if dens < 2: dens = 2

    # edge_density = float(last["BrushSize"])

    undo_total = float(last["TotalUndoCount"])
    rhy = int(round(undo_total))
    if rhy < 0: rhy = 0
    if rhy < 2: rhy = 2

    chroma_value = (max(r_i,g_i,b_i)/255.0) - (min(r_i,g_i,b_i)/255.0)
    chr = int(round(chroma_value))
    if chr < 0: chr = 0
    if chr < 2: chr = 2

    color_name = _classify_color_5(r_i, g_i, b_i)
    key_name = _COLOR_KEY[color_name]
    key = _KEY2IDX[key_name]
    bpm = _BPM_MAP[color_name]

    pos = 0

    return {
        "key_idx": key,
        "bpm": bpm,
        "mode_major": mode_major,
        "reg": reg,
        "rhy_idx": rhy,
        "dens_idx": dens,
        "chr_idx": chr,
        "pos_idx": pos,
    }
