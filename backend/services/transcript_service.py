class TranscriptService:

    def merge_transcript_and_speakers(
        self,
        transcription_segments,
        diarization
    ):

        combined_segments = []

        for segment in transcription_segments:

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

        return combined_segments