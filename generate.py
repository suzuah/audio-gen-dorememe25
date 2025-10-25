import re, math
from typing import List, Optional
from pathlib import Path
import torch
import torch.nn as nn
import miditoolkit

"""SEED = 42
random.seed(SEED); torch.manual_seed(SEED)"""

NOTE_RE = re.compile(r"^NOTE_(\d+)$")
DUR_RE  = re.compile(r"^DUR_(\d+)$")
VEL_RE  = re.compile(r"^VEL_(\d+)$")
POS_RE  = re.compile(r"^POS_(\d+)$")
BPM_RE  = re.compile(r"^BPM_(\d+)$")
    
def bars_to_seconds(bars: int, bpm: int, beats_per_bar: int = 4) -> float:
    # 총 시간 = 마디 * (한 마디의 초)
    return bars * (beats_per_bar * 60.0 / bpm)

def parse_bpm(prefix_tokens, default=120):
    for t in prefix_tokens:
        m = BPM_RE.match(t)
        if m:
            try: return int(m.group(1)) # (\d+)
            except: pass
    return default
    

@torch.no_grad()
def generate_until_seconds(model: nn.Module,
                           dataset,
                           prefix_tokens: List[str],
                           target_sec: float,
                           temperature = 1.0,
                           top_p: float = 0.98,
                           max_steps: int = 8000,
                           beats_per_bar: int = 4,
                           fill_last_bar: bool = False,
                           generator: Optional[torch.Generator] = None,
                           ):
    model.eval()
    stoi = dataset.stoi
    itos = dataset.itos
    PAD_ID = dataset.PAD_ID
    EOS_ID = dataset.EOS_ID

    bpm = parse_bpm(prefix_tokens, default=120)

    # 목표 마디 수 = 시간(sec) → 마디 변환
    # (초당 박수) / (마디당 박수)
    # (bpm / 60) / beats_per_bar
    # bpm / (60 * beats_per_bar)
    target_bars = max(4, int(math.ceil(target_sec * bpm / (60 * beats_per_bar))))

    dev = next(model.parameters()).device

    # prefix 준비
    ids = [stoi.get(t, PAD_ID) for t in prefix_tokens]
    x = torch.tensor(ids, dtype=torch.long, device=dev).unsqueeze(0)

    # g = torch.Generator(device=dev).manual_seed(seed) if seed is not None else None

    prefix_bars = sum(1 for t in prefix_tokens if t == "BAR")
    bars = prefix_bars

    # 목표 마디 도달 여부
    limit = (bars >= target_bars)

    steps = 0
    stop = False # EOS 후 즉시 종료 여부
    in_last_bar = False # 마지막 마디 진입 여부
    lastbar_note_cnt = 0 # 마지막 마디 노트 개수

    while steps < max_steps:
        steps += 1

        if x.size(1) > dataset.block_size:
            x = x[:, -dataset.block_size:] # 슬라이딩

        logits = model(x, pad_id=PAD_ID)[:, -1, :] # [B, V] == [1, V]
        logits = logits / max(1e-6, temperature)
        base_logits = logits.clone()

        # 목표 마디 도달 전 EOS 금지
        if EOS_ID is not None:
            if fill_last_bar: # 마지막 마디 시작 → 최소 1개 이상의 노트가 등장할 수 있도록 EOS 지연
                forbid_eos = (not limit) or (in_last_bar and lastbar_note_cnt == 0)
            else:
                forbid_eos = (not limit)
            if forbid_eos:
                logits[:, EOS_ID] = float("-inf") # EOS 확률 0 처리

        # Nucleus(Top-p) Sampling
        # 누적 확률이 top_p를 넘기 전까지 후보 유지
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits[0], dim=-1) # [V] 확률 분포
        cum = torch.cumsum(probs, dim=-1) # 누적합
        cutoff_idx = (cum > top_p).nonzero(as_tuple=False) # cum > top_p가 True인 위치들
        cutoff = (int(cutoff_idx[0].item()) + 1) if cutoff_idx.numel() > 0 else probs.size(0)
        if cutoff < 1:
            cutoff = 1

        keep = torch.zeros_like(logits, dtype=torch.bool) # [1, V] False 초기화
        keep.scatter_(1, sorted_idx[:, :cutoff], True) # dim=1 기준 상위 cutoff개 위치에 True
        logits = logits.masked_fill(~keep, float("-inf")) # True↔False, True(기존 False)에 대해 확률 0 처리

        # 모든 로짓이 -inf가 되는 예외 상황
        row = logits[0] # [V]
        if torch.isneginf(row).all():
            logits = base_logits.clone()
            if EOS_ID is not None and not limit:
                logits[:, EOS_ID] = float("-inf")

        # 최종 확률 분포
        probs = torch.softmax(logits, dim=-1)
        if (not torch.isfinite(probs).all()) or (probs.sum() <= 0):
            intnext = int(torch.argmax(logits[0]).item())
            next_id = torch.tensor([[intnext]], dtype=torch.long, device=logits.device)
        else:
            next_id = torch.multinomial(probs, 1, generator=generator)

        nid = int(next_id.item())
        tok = itos[nid]

        if tok == "BAR":
            if limit: # 목표 마디 도달
                stop = True
                continue

        ids.append(nid)
        x = torch.cat([x, next_id], dim=1) # 다음 스텝 입력으로 사용

        if tok == "BAR":
            bars += 1 # 마디 추가
            if bars == target_bars: # 목표 마디 도달
                limit = True
                in_last_bar = True
                
        if in_last_bar and tok.startswith("NOTE_"):
            lastbar_note_cnt += 1

        if stop and tok == "EOS":
            break
        if tok == "EOS":
            break

    toks = [itos[i] for i in ids]
    approx = bars_to_seconds(bars, bpm, beats_per_bar)
    print(f"{approx:.1f}s  (bars={bars}, bpm={bpm})")
    print("Generated tokens:\n", " ".join(toks))
    """print("prefix_bars =", sum(1 for t in prefix_tokens if t == "BAR"))
    print("bars_in_output =", sum(1 for t in toks if t == "BAR"))"""

    return toks


def vbin_to_vel(vbin: int, vel_bins: int = 8) -> int:
    if vbin is None:
        vbin = vel_bins
    vbin = max(1, min(vel_bins, int(vbin)))
    step = 127 / vel_bins
    vel = int(round((vbin - 0.5) * step)) # 중앙값을 MIDI velocity로 매핑
    return max(1, min(127, vel))


def tokens_to_midi(tokens, out_midi_path: str, tpq: int = 480, grid_div: int = 4):
    bpm = 120
    for t in tokens:
        m = BPM_RE.match(t)
        if m:
            bpm = int(m.group(1))
            break

    midi = miditoolkit.MidiFile()
    midi.ticks_per_beat = tpq # 한 박자(1/4음표)당 tick 수
    midi.tempo_changes = [miditoolkit.TempoChange(bpm, time=0)] # 시작 시점 템포 이벤트 등록

    inst = miditoolkit.Instrument(program=0, is_drum=False, name="melody") # piano
    midi.instruments = [inst]
    
    total_bars = sum(1 for t in tokens if t == "BAR")
    bar_ticks = tpq * 4 # 한 마디당 tick 수
    grid_ticks = tpq // grid_div # 한 그리드당 tick 수
    max_tick = (total_bars + 1) * bar_ticks

    cur_bar = 0 # 마디
    cur_pos = 0 # 마디 내 그리드 위치
    pending = None
    last_end_tick = 0

    def flush_note():
        nonlocal pending, last_end_tick
        if pending and pending.get("dur") is not None:
            start = max(0, cur_bar * bar_ticks + pending["pos"] * grid_ticks)
            dur_ticks = max(grid_ticks, pending["dur"] * grid_ticks)
            
            start = max(start, last_end_tick)
            end = min(start + dur_ticks, max_tick)

            if end <= start:
                pending = None
                return

            vel = vbin_to_vel(pending.get("velbin", 8))

            # 트랙에 추가
            inst.notes.append(miditoolkit.Note(
                velocity=vel,
                pitch=pending["pitch"],
                start=start,
                end=end
            ))

            last_end_tick = end
            pending = None # 초기화


    for t in tokens:

        if t == "BAR":
            cur_bar += 1
            cur_pos = 0
            continue

        m = POS_RE.match(t)
        if m:
            cur_pos = int(m.group(1))
            continue

        m = NOTE_RE.match(t)
        if m:
            pending = {
                "pitch": int(m.group(1)),
                "dur": None,
                "velbin": None,
                "pos": cur_pos
            }
            continue

        m = DUR_RE.match(t)
        if m and pending:
            pending["dur"] = int(m.group(1))
            if pending.get("velbin") is not None:
                flush_note()
            continue

        m = VEL_RE.match(t)
        if m and pending:
            pending["velbin"] = int(m.group(1))
            if pending.get("dur") is not None:
                flush_note()
            continue

        if t == "EOS":
            break

    flush_note()

    Path(out_midi_path).parent.mkdir(parents=True, exist_ok=True)
    midi.dump(out_midi_path)
    return out_midi_path