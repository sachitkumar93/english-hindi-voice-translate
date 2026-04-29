import webrtcvad
from config import SAMPLE_RATE, FRAME_DURATION_MS, SILENCE_THRESHOLD_MS, MAX_UTTERANCE_MS

vad = webrtcvad.Vad()
vad.set_mode(2)

def is_speech(frame: bytes) -> bool:
    try:
        return vad.is_speech(frame, SAMPLE_RATE)
    except Exception:
        return False

class AudioBufferManager:
    def __init__(self):
        self.utterance_buffer = bytearray()
        self.silence_ms = 0
        self.on_segment_ready = None

    def process_frame(self, frame: bytes, speech: bool):
        if speech:
            self.silence_ms = 0
            self.utterance_buffer.extend(frame)
        else:
            if self.utterance_buffer:
                self.silence_ms += FRAME_DURATION_MS
                if self.silence_ms >= SILENCE_THRESHOLD_MS:
                    self._flush()

        duration_ms = len(self.utterance_buffer) / (SAMPLE_RATE * 2) * 1000
        if duration_ms >= MAX_UTTERANCE_MS:
            self._flush()

    def _flush(self):
        if self.utterance_buffer and self.on_segment_ready:
            self.on_segment_ready(bytes(self.utterance_buffer))
        self.utterance_buffer.clear()
        self.silence_ms = 0
