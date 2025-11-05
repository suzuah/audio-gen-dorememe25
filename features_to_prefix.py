from typing import List, Dict
from pathlib import Path
import pandas as pd
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

    # KEY
    r = r_i / 255.0
    g = g_i / 255.0
    b = b_i / 255.0
    yb = (r + g) / 2.0 - b # yellow - blue
    rg = r - g
    angle = math.atan2(rg, yb) # opponent color space
    if angle < 0:
        angle += 2 * math.pi
    sector = int(round(angle / (2 * math.pi / 12))) % 12
    KEY_NAMES = ["D","A","E","B","F#","C#","G#","D#","A#","F","C","G"] # 스크랴 빈이 색-KEY
    key_name = KEY_NAMES[sector]
    _KEYS = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    _KEY2IDX = {k:i for i,k in enumerate(_KEYS)}
    key = _KEY2IDX[key_name]

    # MODE
    if season in {"spring", "summer"}:
        mode_major = True
    else:
        mode_major = False

    # BPM
    brightness = 0.299 * r_i + 0.587 * g_i + 0.114 * b_i # Luma, 표준 밝기 계산식
    brightness_norm = (brightness - 30) / (230 - 30) # normalization
    brightness_norm = max(0.0, min(1.0, brightness_norm)) # clamp
    if brightness_norm < 0.2:
        bpm = 60
    elif brightness_norm < 0.4:
        bpm = 80
    elif brightness_norm < 0.6:
        bpm = 100
    elif brightness_norm < 0.8:
        bpm = 120
    else:
        bpm = 140

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
    actions = undo_total + stroke_count
    if actions <= 0:
        ratio = 0.0
    else:
        ratio = undo_total / actions # 전체 actions 중 undo 비율
    rhy = int(round(2.0 * ratio))
    rhy = max(0, min(2, rhy))

    # DENS
    THIN  = 0.30
    THICK = 0.70
    if "BrushSize" in df_sorted.columns:
        brush_size = float(df_sorted.iloc[-1]["BrushSize"]) # 보류
        if brush_size < THIN:
            dens = 0
        elif brush_size < THICK:
            dens = 1
        else:
            dens = 2

    # CHR
    TRANSPARENT = 0.40
    OPAQUE = 0.60
    if "ColorA" in df_sorted.columns:
        alpha = float(df_sorted["ColorA"].median())
        if alpha < TRANSPARENT:
            chr = 0
        elif alpha > OPAQUE:
            chr = 2
        else:
            chr = 1

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
