import os
import subprocess
import tempfile


class AudioService:

    def convert_to_wav(self, input_path: str) -> str:

        output_path = tempfile.mktemp(suffix=".wav")

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i", input_path,
                    "-ac", "1",
                    "-ar", "16000",
                    output_path
                ],
                check=True,
                capture_output=True,
                text=True
            )

            return output_path

        except subprocess.CalledProcessError as error:
            if os.path.exists(output_path):
                os.remove(output_path)

            raise RuntimeError(
                f"Audio conversion failed: {error.stderr}"
            )