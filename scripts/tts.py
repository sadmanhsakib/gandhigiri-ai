import os, sys, time
import uuid
import torch
from pathlib import Path

# Point to local cloned repo
BASE_DIR = Path(__file__).parent.parent
print(BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "OpenVoice"))
sys.path.insert(0, os.path.join(BASE_DIR, "MeloTTS"))

from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
CKPT_CONVERTER = os.path.join(BASE_DIR, "checkpoints_v2", "converter")
BASE_SE_PATH = os.path.join(
    BASE_DIR, "checkpoints_v2", "base_speakers", "ses", "en-india.pth"
)
REFERENCE_AUDIO = os.path.join(BASE_DIR, "reference_audio", "gandhi_reference.wav")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
TMP_DIR = os.path.join(BASE_DIR, "tmp")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# Validate required files exist before loading
if not os.path.exists(REFERENCE_AUDIO):
    raise FileNotFoundError(
        f"Gandhi reference audio not found at {REFERENCE_AUDIO}\n"
        f"Place your cleaned gandhi_reference.wav inside the reference_audio/ folder."
    )

if not os.path.exists(CKPT_CONVERTER):
    raise FileNotFoundError(
        f"Checkpoints not found at {CKPT_CONVERTER}\n"
        f"Run the snapshot_download script first to download model weights."
    )

# Load models once at startup
print("[TTS] Loading ToneColorConverter...")
tone_color_converter = ToneColorConverter(
    os.path.join(CKPT_CONVERTER, "config.json"), device=DEVICE
)
tone_color_converter.load_ckpt(os.path.join(CKPT_CONVERTER, "checkpoint.pth"))

print("[TTS] Loading MeloTTS...")
tts_model = TTS(language="EN", device=DEVICE)
speaker_ids = tts_model.hps.data.spk2id

# Load base speaker embedding
print("[TTS] Loading base speaker embedding (en-india)...")
source_se = torch.load(BASE_SE_PATH, map_location=DEVICE).to(DEVICE)

# Extract Gandhi voice style from reference audio
print("[TTS] Extracting Gandhi voice style embedding...")
target_se, _ = se_extractor.get_se(
    REFERENCE_AUDIO, tone_color_converter, target_dir=TMP_DIR, vad=True
)

print("[TTS] Ready.")


def main():
    test_text = (
        "Beta, sach bolo aur sir uthake jio."
    )
    print(f"[TEST] Synthesizing: {test_text}")
    path = synthesize(test_text)
    print(f"[TEST] Output: {path}")


def synthesize(text: str) -> str:
    """
    Takes a text string.
    Returns the file path of the generated Gandhi-voiced .wav file.
    """
    if not text or text.strip() == "":
        raise ValueError("Cannot synthesize empty text.")

    unique_id = uuid.uuid4().hex[:8]
    tmp_path = os.path.join(TMP_DIR, f"base_{unique_id}.wav")
    output_path = os.path.join(OUTPUT_DIR, f"gandhi_{unique_id}.wav")

    # Step 1 — Generate base speech with MeloTTS
    print("[TTS] Generating base speech...")
    tts_model.tts_to_file(
        text=text,
        speaker_id=speaker_ids["EN_INDIA"],  # Indian English accent as base
        output_path=tmp_path,
        speed=0.82,
    )

    # Step 2 — Apply Gandhi tone color
    print("[TTS] Applying Gandhi voice style...")
    tone_color_converter.convert(
        audio_src_path=tmp_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=output_path,
        message="@MyShell",
    )

    # Clean up tmp file
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    print(f"[TTS] Audio saved to {output_path}")
    return output_path


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\n✅ Execution completed in {time.time() - start_time:.2f} seconds")
