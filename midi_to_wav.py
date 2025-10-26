import subprocess
from pathlib import Path

def midi_to_wav(midi_path, wav_path, sf2_path, sample_rate=32000):
    midi_path = str(midi_path)
    wav_path  = str(wav_path)
    sf2_path  = str(sf2_path)

    Path(wav_path).parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "fluidsynth",
        "-ni",
        "-F", str(wav_path),
        "-r", str(sample_rate),
        str(sf2_path),
        str(midi_path),
    ]

    subprocess.run(cmd, check=True)
    return wav_path