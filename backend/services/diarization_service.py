import os
import torch
import soundfile as sf

from scipy.signal import resample_poly
from pyannote.audio import Pipeline


class DiarizationService:

    def __init__(self):
        print("Loading speaker diarization model...")

        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            token=os.environ["HF_TOKEN"]
        )

        self.pipeline.to(torch.device("cpu"))

        print("Pyannote loaded!")

    def identify_speakers(self, audio_path: str):

        waveform, sample_rate = sf.read(
            audio_path,
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

        result = self.pipeline({
            "waveform": waveform,
            "sample_rate": sample_rate
        })

        return result.speaker_diarization