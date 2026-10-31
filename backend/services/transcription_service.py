import whisper


class TranscriptionService:

    def __init__(self):
        print("Loading Whisper model...")
        self.model = whisper.load_model("base")
        print("Whisper model loaded!")

    def transcribe(self, audio_path: str):
        result = self.model.transcribe(
            audio_path,
            task="transcribe"
        )

        return {
            "language": result["language"],
            "transcript": result["text"],
            "segments": result["segments"]
        }