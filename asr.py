import numpy as np
from faster_whisper import WhisperModel

print("Loading Whisper model... (downloads ~150MB on first run)")
model = WhisperModel("base", device="cpu", compute_type="int8")
print("Whisper ready.")

def transcribe(audio_bytes: bytes) -> dict:
    audio_np = (
        np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    )

    segments, info = model.transcribe(
        audio_np,
        language="en",
        vad_filter=True,
        word_timestamps=True,
        condition_on_previous_text=False,
        beam_size=1,
        best_of=1,
        initial_prompt="Technical discussion: pull request, push, commit, GitHub, Docker, Kubernetes, AWS, merge, branch, deploy, pipeline, API, backend, frontend, standup, EOB, ETA, sprint, Jira, Slack, Google Meet",
    )

    full_text = ""
    scores = []
    for segment in segments:
        full_text += segment.text
        for word in segment.words:
            scores.append(word.probability)

    avg_confidence = sum(scores) / len(scores) if scores else 0.0
    return {"text": full_text.strip(), "confidence": avg_confidence}
