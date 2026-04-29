import asyncio
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
import io

# ──────────────────────────────────────────────────────
# playback.py — Audio playback with barge-in support
#
# Manages a queue of TTS audio chunks and plays them
# sequentially. Supports barge-in — if the user speaks
# mid-playback, interrupt() cuts the audio immediately
# without killing the microphone input stream.
#
# Key design: uses a dedicated sd.OutputStream instead
# of sd.play() so we can stop output independently
# of the microphone sd.InputStream.
# ──────────────────────────────────────────────────────

class PlaybackQueue:
    def __init__(self):
        self.queue = None
        self.is_playing = False
        self._output_stream = None
        self._finished_event = None

    def init_queue(self):
        # Must be called after the event loop starts
        self.queue = asyncio.Queue()

    def interrupt(self):
        """
        Immediately stops current playback and clears queue.
        Called when VAD detects user speaking during TTS output.
        
        Also sets _finished_event so the await in run() unblocks
        cleanly instead of hanging after the stream is force-closed.
        """
        self.is_playing = False

        if self._output_stream is not None:
            try:
                self._output_stream.stop()
                self._output_stream.close()
            except Exception:
                pass
            self._output_stream = None

        if self._finished_event is not None:
            try:
                self._finished_event.set()
            except Exception:
                pass

        if self.queue:
            while not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except Exception:
                    break

    async def enqueue(self, audio_bytes: bytes):
        """Add MP3 audio bytes to the playback queue."""
        await self.queue.put(audio_bytes)

    async def run(self):
        """
        Background worker — runs forever, plays audio as it arrives.
        
        MP3 → PCM conversion:
          edge-tts outputs MP3
          sounddevice needs float32 PCM
          pydub handles the conversion + resampling to 44100Hz
          (Galaxy Buds and most Bluetooth devices need 44100Hz)
        """
        self.init_queue()
        loop = asyncio.get_event_loop()

        while True:
            audio_bytes = await self.queue.get()
            self.is_playing = True

            try:
                # Decode MP3 and resample to 44100Hz for Bluetooth compatibility
                seg = AudioSegment.from_mp3(io.BytesIO(audio_bytes))
                seg = seg.set_frame_rate(44100).set_channels(1).set_sample_width(2)
                pcm = np.frombuffer(seg.raw_data, dtype=np.int16).astype(np.float32) / 32767.0

                finished = asyncio.Event()
                self._finished_event = finished

                def playback_callback(outdata, frames, time, status):
                    nonlocal pcm
                    chunk = pcm[:frames]
                    if len(chunk) < frames:
                        # End of audio — pad with silence and stop
                        outdata[:len(chunk), 0] = chunk
                        outdata[len(chunk):, 0] = 0
                        raise sd.CallbackStop()
                    else:
                        outdata[:, 0] = chunk
                        pcm = pcm[frames:]

                def on_finished():
                    # Called by sounddevice when stream ends naturally
                    # or after CallbackStop — unblocks await finished.wait()
                    loop.call_soon_threadsafe(finished.set)

                self._output_stream = sd.OutputStream(
                    samplerate=44100,
                    channels=1,
                    dtype='float32',
                    device=sd.default.device[1],  # uses whatever output device is set
                    callback=playback_callback,
                    finished_callback=on_finished,
                )
                self._output_stream.start()
                await finished.wait()

            except Exception as e:
                if "CallbackStop" not in str(type(e).__name__):
                    print(f"[Playback error] {e}")
            finally:
                if self._output_stream:
                    try:
                        self._output_stream.close()
                    except Exception:
                        pass
                    self._output_stream = None
                self._finished_event = None
                self.is_playing = False
