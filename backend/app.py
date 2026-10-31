
import os
import tempfile
import torch
import whisper
import soundfile as sf

from scipy.signal import resample_poly
from fastapi import FastAPI, UploadFile, File
from pyannote.audio import Pipeline

app = FastAPI()

# 1. Load Whisper
print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper loaded!")

# 2. Load Pyannote
print("Loading speaker diarization model...")

diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-community-1",
    token=os.environ["HF_TOKEN"]
)

diarization_pipeline.to(torch.device("cpu"))

print("Pyannote loaded!")


@app.get("/")
def home():
    return {
        "message": "Medical Audio Assistant backend is running!"
    }


@app.post("/process-audio")
async def process_audio(audio: UploadFile = File(...)):

    # Save uploaded audio temporarily
    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix
    ) as temp_audio:
        temp_audio.write(await audio.read())
        temp_path = temp_audio.name

    try:
        # 3. Transcribe with Whisper
        print("Transcribing audio...")

        transcription = whisper_model.transcribe(
            temp_path,
            task="transcribe"
        )

        # 4. Load audio for Pyannote
        print("Identifying speakers...")

        waveform, sample_rate = sf.read(
            temp_path,
            dtype="float32",
            always_2d=True
        )

        waveform = waveform.mean(axis=1)

        if sample_rate != 16000:
            waveform = resample_poly(
                waveform, 16000, sample_rate
            ).astype("float32")
            sample_rate = 16000

        waveform = torch.from_numpy(waveform).unsqueeze(0)

        diarization_result = diarization_pipeline({
            "waveform": waveform,
            "sample_rate": sample_rate
        })

        diarization = diarization_result.speaker_diarization

        # 5. Match transcript segments with speakers
        combined_segments = []

        for segment in transcription["segments"]:

            start = segment["start"]
            end = segment["end"]

            speaker_times = {}

            for turn, _, speaker in diarization.itertracks(
                yield_label=True
            ):
                overlap = max(
                    0,
                    min(end, turn.end) - max(start, turn.start)
                )

                speaker_times[speaker] = (
                    speaker_times.get(speaker, 0) + overlap
                )

            speaker = (
                max(speaker_times, key=speaker_times.get)
                if speaker_times and max(speaker_times.values()) > 0
                else "UNKNOWN"
            )

            combined_segments.append({
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": segment["text"].strip()
            })

        return {
            "language": transcription["language"],
            "transcript": transcription["text"],
            "segments": combined_segments
        }

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
