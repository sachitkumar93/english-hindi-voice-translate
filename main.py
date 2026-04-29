import asyncio
import numpy as np
import sounddevice as sd
import time

from config import SAMPLE_RATE, FRAME_SIZE, MIN_CONFIDENCE
from audio_buffer import AudioBufferManager, is_speech
from asr import transcribe
from translator import translate
from tts import synthesize
from playback import PlaybackQueue

# main.py — Pipeline orchestrator
#
# Wires all components together:
#   MIC → VAD → Buffer → ASR → Translator → TTS → Speaker
#
# The core challenge: sounddevice audio callback runs in
# a separate OS thread, but ASR/translation/TTS are async.
# Bridge: asyncio.run_coroutine_threadsafe() schedules
# async work from the audio thread onto the event loop.

# Set audio devices:
# device 2 = MacBook Air Microphone (input)
# device 1 = Galaxy Buds (output)
# change for your sound device settings:
# Run: python3 -c "import sounddevice; print(sounddevice.query_devices())"

sd.default.device = (2, 1)

playback = PlaybackQueue()
buffer_manager = AudioBufferManager()
main_loop = None


async def handle_segment(audio_bytes: bytes):
    start = time.time()

    if playback.is_playing:
        print("[Barge-in]")
        playback.interrupt()

    # ASR
    result = transcribe(audio_bytes)
    print(f" ASR time: {(time.time()-start)*1000:.0f}ms")

    text = result["text"]
    confidence = result["confidence"]

    if not text:
        return
    if confidence < MIN_CONFIDENCE:
        print(f"[Low confidence {confidence:.2f} — skipped] {text}")
        return

    print(f"\n EN: {text}")

    # Translation
    t2 = time.time()
    hindi = await translate(text)
    print(f" Translation time: {(time.time()-t2)*1000:.0f}ms")

    if not hindi:
        return
    print(f"HI: {hindi}")

    # TTS
    t3 = time.time()
    audio = await synthesize(hindi)
    print(f"TTS: {(time.time()-t3)*1000:.0f}ms")

    elapsed = (time.time() - start) * 1000
    print(f"Playing... Total: {elapsed:.0f}ms")

    await playback.enqueue(audio)


def on_segment_ready(audio_bytes: bytes):
    """
    Bridge from audio thread → async event loop.
    sounddevice callback is synchronous, pipeline is async.
    """
    if main_loop:
        asyncio.run_coroutine_threadsafe(handle_segment(audio_bytes), main_loop)


async def main():
    global main_loop
    main_loop = asyncio.get_event_loop()

    buffer_manager.on_segment_ready = on_segment_ready
    asyncio.create_task(playback.run())

    print("\n Pipeline ready. Speak in English...\n")
    print("Ctrl+C to stop.\n")

    def audio_callback(indata, frames, time, status):
        """Called every 30ms by sounddevice with a fresh audio chunk."""
        frame = (indata[:, 0] * 32767).astype(np.int16).tobytes()
        buffer_manager.process_frame(frame, is_speech(frame))

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=FRAME_SIZE,
        device=2,
        callback=audio_callback,
    ):
        await asyncio.sleep(float("inf"))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nStopped.")
