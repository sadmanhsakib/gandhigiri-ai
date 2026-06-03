import os, time
from dotenv import load_dotenv
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.database.util import load_rttm
from pydub import AudioSegment

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
AUDIO_FILE_NAME = "reference_audio/laghe_raho_munna_bhai.wav"
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=HF_TOKEN,
)


def main():
    diarization = get_audio_segments()
    
    rttm_data = load_rttm("reference_audio/refaudio.rttm")
    file_key = list(rttm_data.keys())[0]
    diarization = rttm_data[file_key]
    
    get_prefered_audio_clips(diarization, speaker="SPEAKER_08")


def get_audio_segments():
    with ProgressHook() as hook:
        raw_output = pipeline(AUDIO_FILE_NAME, hook=hook)

    # Extract the real Annotation from the wrapper
    diarization = raw_output.speaker_diarization

    with open("reference_audio/refaudio.rttm", "w") as rttm:
        diarization.write_rttm(rttm)

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"{speaker}: {turn.start:.1f}s → {turn.end:.1f}s")

    return diarization


def get_prefered_audio_clips(diarization, speaker: str):
    audio = AudioSegment.from_wav(AUDIO_FILE_NAME)
    clips = []

    for segment, _, spk in diarization.itertracks(yield_label=True):
        if spk == speaker:
            clip = audio[int(segment.start * 1000):int(segment.end * 1000)]
            clips.append(clip)

    if not clips:
        print(f"⚠️ No audio segments found for speaker: {speaker}")
        return

    combined = sum(clips)
    combined.export("reference_audio/gandhi_reference.wav", format="wav")
    print(f"✅ Exported {speaker} segments to reference_audio/gandhi_reference.wav")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"\n✅ Execution completed in {time.time() - start_time:.2f} seconds")
