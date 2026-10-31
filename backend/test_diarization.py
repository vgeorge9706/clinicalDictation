
# import os
# import torch
# import soundfile as sf
# from scipy.signal import resample_poly
# from pyannote.audio import Pipeline

# print("Loading speaker diarization model...")

# pipeline = Pipeline.from_pretrained(
#     "pyannote/speaker-diarization-community-1",
#     token=os.environ["HF_TOKEN"]
# )

# pipeline.to(torch.device("cpu"))

# print("Model loaded successfully!")

# audio_path = "sample-audio.wav"

# print("Reading audio...")

# audio, sample_rate = sf.read(
#     audio_path,
#     dtype="float32",
#     always_2d=True
# )

# # Convert stereo to mono
# audio = audio.mean(axis=1)

# # Resample to 16 kHz
# if sample_rate != 16000:
#     audio = resample_poly(
#         audio, 16000, sample_rate
#     ).astype("float32")
#     sample_rate = 16000

# # Create waveform tensor
# waveform = torch.from_numpy(audio).unsqueeze(0)

# print("Analysing audio...")

# result = pipeline(
#     {
#         "waveform": waveform,
#         "sample_rate": sample_rate
#     },
#     num_speakers=3
# )

# diarization = result.speaker_diarization

# print("\nSpeaker results:")

# for turn, _, speaker in diarization.itertracks(yield_label=True):
#     print(
#         f"{speaker}: "
#         f"{turn.start:.2f}s - {turn.end:.2f}s"
#     )


from services.medical_ai_service import MedicalAIService

ai = MedicalAIService()

transcript = "\"The patient is 45 years old, sorry, 54. The cyst is 2 mm, actually 2 cm.\""

result = ai.correct_transcript(transcript)

print("Corrected transcript:")
print(result)