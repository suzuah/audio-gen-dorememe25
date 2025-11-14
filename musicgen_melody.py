import os, re, random, time
from pathlib import Path
import numpy as np
import torch
import soundfile as sf
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration

MODEL_ID = "facebook/musicgen-melody"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROCESSOR = None
MODEL = None

def init_musicgen(device: str | None = None, use_fp16: bool = True):
    global PROCESSOR, MODEL, DEVICE
    if device is None:
        device = DEVICE
    if PROCESSOR is None or MODEL is None or device != DEVICE:
        DEVICE = device
        PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID)
        dtype = torch.float16 if (use_fp16 and device == "cuda") else None
        MODEL = MusicgenMelodyForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=dtype
        ).to(device)
        MODEL.eval()

def prefix_to_text(prefix, include_tokens=False, season=None, variety=True):
    """
    prefix: ["KEY_7","MODE_MAJ","BPM_120","REG_MID","RHY_2","DENS_2","CHR_1"]
    returns: natural-language prompt string for MusicGen
    """
    rng = random.Random(time.time_ns())

    if prefix is None:
        prefix = []
    if isinstance(prefix, str):
        tokens = [t for t in re.split(r"[,\s]+", prefix) if "_" in t]
    else:
        tokens = [str(t) for t in prefix if isinstance(t, (str, bytes))]

    d = {}
    for p in tokens:
        if "_" in p:
            k, v = p.split("_", 1)
            d[k] = v

    # KEY
    try:
        key_pc = int(d.get("KEY", 0))
    except ValueError:
        key_pc = 0
    key_names = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]
    key_name = key_names[key_pc % 12]

    # MODE
    mode = d.get("MODE", "MAJ")
    mode_name = "major" if mode == "MAJ" else "minor"
    mode_desc = "major-key tonality" if mode == "MAJ" else "minor-key tonality"

    # BPM
    try:
        bpm = int(d.get("BPM", 120))
    except ValueError:
        bpm = 120

    if bpm == 60:
        bpm_desc = "slow tempo"
    elif bpm == 80:
        bpm_desc = "slow to moderate tempo"
    elif bpm == 100:
        bpm_desc = "moderate tempo"
    elif bpm == 120:
        bpm_desc = "fast tempo"
    else:
        bpm_desc = "very fast tempo"

    # REG
    reg = d.get("REG", "MID")
    reg_map = {"LOW": "focused on lower register", "MID": "mid-range voicing", "HIGH": "upper register voicing"}
    reg_txt = reg_map.get(reg, "mid-range voicing")

    # RHY
    try:
        rhy = max(0, min(2, int(d.get("RHY", 1))))
    except ValueError:
        rhy = 1
    rhy_map = ["straight feel", "light syncopation", "marked syncopation"]
    rhy_txt = rhy_map[rhy]

    # DENS
    try:
        dens = max(0, min(2, int(d.get("DENS", 1))))
    except ValueError:
        dens = 1
    dens_map = ["sparse phrasing with space", "moderate note density", "busy lines with constant motion"]
    dens_txt = dens_map[dens]

    # CHR
    try:
        ch = max(0, min(2, int(d.get("CHR", 1))))
    except ValueError:
        ch = 1
    chr_map = ["mostly diatonic lines", "occasional chromatic passing tones", "frequent chromatic runs"]
    chr_txt = chr_map[ch]

    # Season
    season_txt = None
    if isinstance(season, str):
        s = season.strip().lower()
        season_map = {
            "spring": "music suited for spring-like: bright, airy, hopeful mood", # 왈츠
            "summer": "music suited for summer-like: vibrant, open, expansive mood", # 트로피칼
            "autumn": "music suited for autumn-like: warm, mellow, reflective mood", # 재즈
            "winter": "music suited for winter-like: calm, crystalline, sparse mood" # 크리스마스 캐롤
        }
        season_txt = season_map.get(s)
        
        style_map = {
            "spring": "spring-inspired piano waltz feel in 3/4 time",
            "summer": "summer-style tropical groove led by piano",
            "autumn": "autumn-style jazz or ballad piano texture",
            "winter": "winter-style Christmas carol feeling with piano focus"
        }
        style_txt = style_map.get(s)

    if variety:
        lead_piano_pool = [
            "instrumental piece featuring piano",
            "melody-driven instrumental",
            "expressive piano performance",
            "cinematic piano cue",
            "lively piano groove"
        ]
        lead_piano = rng.choice(lead_piano_pool)

    # 최종 문장
    parts = [
        style_txt,
        season_txt,
        f"in {key_name} {mode_name}",
        mode_desc,
        f"around {bpm} bpm",
        bpm_desc,
        f"{reg_txt}",
        f"{rhy_txt}",
        f"{dens_txt}",
        f"{chr_txt}",
        "melody-forward",
        "no vocals",
        "light arrangement",
    ]

    base = ", ".join(parts)

    if include_tokens:
        tok = " ".join([
            f"KEY_{key_pc}",
            f"MODE_{'MAJ' if mode_name=='major' else 'MIN'}",
            f"BPM_{bpm}",
            f"REG_{reg}",
            f"RHY_{rhy}",
            f"DENS_{dens}",
            f"CHR_{ch}"
        ])
        if isinstance (season, str) and season.strip():
            tok += f" SEASON_{season.strip().upper()}"
        print(f"(tokens: {tok})")

    return base

def stylize_melody(
    melody_wav_path: str,
    out_wav_path: str,
    prompt: str,
    *,
    device: str | None = None,
    use_fp16: bool = True,
    do_sample: bool = True,
    temperature: float = 1.05,
    top_k: int = 250,
    top_p: float = 0.92,
    guidance_scale: float = 3.0,
    max_new_tokens: int = 512,
    ) -> str:
    init_musicgen(device=device, use_fp16=use_fp16)

    audio, sr = sf.read(melody_wav_path)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = torch.as_tensor(audio, dtype=torch.float32)

    inputs = PROCESSOR(
        audio=audio,
        sampling_rate=sr,
        text=prompt,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        audio_values = MODEL.generate(
            **inputs,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            guidance_scale=guidance_scale,
            max_new_tokens=max_new_tokens
        )

    out_sr = MODEL.config.audio_encoder.sampling_rate
    Path(out_wav_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav_path, audio_values[0, 0].cpu().numpy(), out_sr)
    return out_wav_path

