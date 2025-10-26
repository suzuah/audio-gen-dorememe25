import re, os, random
from pathlib import Path
import torch
import pretty_midi

from features_to_prefix import read_csv_strict, build_prefix_tokens, session_to_prefix
from load_model import load_model
from generate import generate_until_seconds, tokens_to_midi
from midi_to_wav import midi_to_wav
from musicgen_melody import init_musicgen, prefix_to_text, stylize_melody

DATA_JSONL = "./data/melody_tok.jsonl"
VOCAB_JSON = "./data/melody_voc.json"
CKPT_PATH  = "./ckpt/melModel_tf.pt"

INPUT_CSV = "./data/sample2.csv"
TARGET_RAW_IDX = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

class Cfg:
    def __init__(self):
        self.block_size = 384
        self.hidden_size = 384
        self.num_heads = 6
        self.num_layers = 8
        self.ffn_hidden_size = 4 * self.hidden_size
        self.dropout = 0.1
        self.batch_size = 4
        self.lr = 3e-4
        self.steps = 3000
        self.print_every = 100
        self.save_every = 1000

cfg = Cfg()

# Load
model, dataset = load_model(CKPT_PATH, DATA_JSONL, VOCAB_JSON, cfg, DEVICE)

# Generate melody by prefix
def generate(prefix_tokens, target_sec=20.0, temperature=1.0, top_p=0.95):
    g = torch.Generator(device=DEVICE).manual_seed(SEED)

    toks = generate_until_seconds(
        model,
        dataset,
        prefix_tokens=prefix_tokens,
        target_sec=target_sec,
        temperature=temperature,
        top_p=top_p,
        generator=g
    )
    return toks

def get_run_dir(base_dir: str = "./runs") -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    run_nums = []
    for child in base.iterdir():
        if child.is_dir():
            m = re.fullmatch(r"(\d+)", child.name)
            if m:
                run_nums.append(int(m.group(1)))

    next_idx = (max(run_nums) + 1) if run_nums else 1
    run_dir = base / str(next_idx)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


if __name__ == "__main__":
    df = read_csv_strict(INPUT_CSV)
    features = session_to_prefix(df)
    prefix = build_prefix_tokens(features)

    print("========== GENERATED PREFIX TOKENS ==========")
    print(prefix)
    print("=============================================")

    toks = generate(prefix)

    runs_dir = get_run_dir("./runs")

    out_midi = runs_dir / f"melody.mid"
    base_wav = runs_dir / f"melody.wav"
    final_wav = runs_dir / f"final.wav"

    # Melody MIDI
    tokens_to_midi(toks, str(out_midi))

    # Melody WAV
    sf2 = os.path.join(os.path.dirname(pretty_midi.__file__), "TimGM6mb.sf2")
    midi_to_wav(str(out_midi), str(base_wav), sf2)

    print("Melody(MIDI) saved:", out_midi)
    print("Melody(WAV) saved:", base_wav)

    init_musicgen(device=DEVICE, use_fp16=True)
    prompt = prefix_to_text(prefix)

    print("========== PROMPT FOR MUSICGEN ==========")
    print(prompt)
    print("=============================================")

    stylize_melody(
        melody_wav_path=str(base_wav),
        out_wav_path=str(final_wav),
        prompt=prompt,
        device=DEVICE,
        use_fp16=True,
        do_sample=True,
        temperature=1.05,
        top_p=0.95,
        max_new_tokens=512
    )
    print("FINAL saved:", final_wav)
    print("Done.")