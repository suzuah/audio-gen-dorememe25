import os, re
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

def prefix_to_text(prefix, include_tokens=False):
    """
    prefix: ["KEY_7","MODE_MAJ","BPM_120","REG_MID","RHY_2","DENS_2","CHR_1"]
    returns: natural-language prompt string for MusicGen
    """
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

    # BPM
    try:
        bpm = int(d.get("BPM", 120))
    except ValueError:
        bpm = 120

    # REG
    reg = d.get("REG", "MID")
    reg_map = {"LOW": "lower register", "MID": "middle register", "HIGH": "upper register"}
    reg_txt = reg_map.get(reg, "middle register")

    # RHY
    try:
        rhy = max(0, min(2, int(d.get("RHY", 1))))
    except ValueError:
        rhy = 1
    rhy_map = ["straight rhythm", "moderate syncopation", "strong syncopation"]
    rhy_txt = rhy_map[rhy]

    # DENS
    try:
        dens = max(0, min(2, int(d.get("DENS", 1))))
    except ValueError:
        dens = 1
    dens_map = ["sparse note density", "moderate note density", "busy note density"]
    dens_txt = dens_map[dens]

    # CHR
    try:
        ch = max(0, min(2, int(d.get("CHR", 1))))
    except ValueError:
        ch = 1
    chr_map = ["minimal chromatic motion", "moderate chromatic motion", "pronounced chromatic motion"]
    chr_txt = chr_map[ch]

    # 최종 문장
    parts = (
        f"solo piano melody",
        f"key: {key_name} {mode_name}",
        f"tempo: {bpm} BPM",
        f"register: {reg_txt}",
        f"rhythm: {rhy_txt}",
        f"density: {dens_txt}",
        f"melodic movement: {chr_txt}",
        "melody-forward, avoid heavy accompaniment",
    )

    base = "; ".join(parts)

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
        base = f"{base} (tokens: {tok})"

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
    top_p: float = 0.95,
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
            top_p=top_p,
            max_new_tokens=max_new_tokens # 약 10s
        )

    out_sr = MODEL.config.audio_encoder.sampling_rate
    Path(out_wav_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav_path, audio_values[0, 0].cpu().numpy(), out_sr)
    return out_wav_path

