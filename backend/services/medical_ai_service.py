
import ollama


class MedicalAIService:

    def __init__(self):
        self.model = "llama3.1:8b"

    def correct_transcript(self, transcript: str):

        prompt = f"""
You are a medical transcription correction assistant.

Your task is to correct only clear, explicit self-corrections
made by the speaker.

Rules:
1. If the speaker explicitly corrects a value, use the corrected value.
2. Do not invent or assume medical information.
3. Preserve uncertainty.
4. Do not change medical terminology unless the correction is clear.
5. Keep the original meaning.
6. Return only the corrected transcript.

Transcript:
{transcript}
"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]
