import edge_tts
import io

VOICE = "hi-IN-SwaraNeural"

async def synthesize(text: str) -> bytes:
    if not text:
        return b""
    
    communicate = edge_tts.Communicate(text, VOICE)
    audio_buffer = io.BytesIO()
    
    # Stream chunks as they arrive instead of waiting for full file
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_buffer.write(chunk["data"])
    
    return audio_buffer.getvalue()