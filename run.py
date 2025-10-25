import os, random
from pathlib import Path
import torch
import pretty_midi
from load_model import load_model
from generate import generate_until_seconds, tokens_to_midi
from midi2wav import midi_to_wav
from musicgen_melody import init_musicgen, prefix_to_text, stylize_melody

DATA_JSONL = "./data/melody_tok.jsonl"
VOCAB_JSON = "./data/melody_voc.json"
CKPT_PATH  = "./runs/melModel_tf.pt"

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

if __name__ == "__main__":
    prefix = ["KEY_7", "MODE_MAJ", "BPM_120", "REG_MID", "RHY_2", "DENS_1", "CHR_1", "BAR", "POS_0"]

    toks = generate(prefix)
    out_midi = Path("./runs/melsamp3.mid")
    base_wav = Path("./runs/melsamp3.wav")
    out_midi.parent.mkdir(parents=True, exist_ok=True)
    base_wav.parent.mkdir(parents=True, exist_ok=True)

    sf2 = os.path.join(os.path.dirname(pretty_midi.__file__), "TimGM6mb.sf2")
    tokens_to_midi(toks, str(out_midi))
    midi_to_wav(str(out_midi), str(base_wav), sf2)

    print("MIDI saved:", out_midi)
    print("WAV saved:", base_wav)

    init_musicgen(device=DEVICE, use_fp16=True)
    prompt = prefix_to_text(prefix)
    print("Prompt:", prompt)

    final_wav = Path("./runs/samp3.wav")
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