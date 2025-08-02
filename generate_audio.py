# generate_audio.py (From google generative AI doc)

import os
import struct  
from google import genai
from google.genai import types
from dotenv import load_dotenv

def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Generates a WAV file header for the given raw audio data."""
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", chunk_size, b"WAVE", b"fmt ", 16, 1,
        num_channels, sample_rate, byte_rate, block_align,
        bits_per_sample, b"data", data_size
    )
    return header + audio_data

def parse_audio_mime_type(mime_type: str) -> dict[str, int]:
    """Parses bits per sample and rate from an audio MIME type string."""
    bits_per_sample = 16
    rate = 24000
    parts = mime_type.split(";")
    for param in parts:
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                pass
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass
    return {"bits_per_sample": bits_per_sample, "rate": rate}

def generate_audio_bytes(api_key: str, text_prompt: str, voice: str = "Zephyr") -> tuple[bytes, str] | None:
    """
    Generates audio and converts it to WAV if the API returns raw PCM data.
    """
    try:
        client = genai.Client(api_key=api_key)
        model_name = "gemini-2.5-pro-preview-tts"
        print(f"Using model: {model_name}")

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=text_prompt)],
            ),
        ]
        generate_content_config = types.GenerateContentConfig(
            response_modalities=["audio"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
                )
            ),
        )

        print("Generating audio stream...")
        response_chunks = client.models.generate_content_stream(
            model=model_name,
            contents=contents,
            config=generate_content_config,
        )

        audio_chunks = []
        mime_type = "audio/mpeg"

        for chunk in response_chunks:
            if (chunk.candidates and chunk.candidates[0].content and
                    chunk.candidates[0].content.parts and
                    chunk.candidates[0].content.parts[0].inline_data and
                    chunk.candidates[0].content.parts[0].inline_data.data):
                
                inline_data = chunk.candidates[0].content.parts[0].inline_data
                if not audio_chunks:
                    mime_type = inline_data.mime_type
                    print(f"Detected audio MIME type: {mime_type}")
                audio_chunks.append(inline_data.data)

        if not audio_chunks:
            print("No audio data received from the API.")
            return None

        full_audio_data = b"".join(audio_chunks)
        print("Audio stream collected successfully.")

        # --- convert audio to wav ---
        if "audio/L16" in mime_type:
            print("Detected raw audio (L16/PCM). Converting to WAV format...")
            full_audio_data = convert_to_wav(full_audio_data, mime_type)
            mime_type = "audio/wav"
            print("Conversion to WAV successful.")

        return (full_audio_data, mime_type)

    except Exception as e:
        print(f"An error occurred during audio generation: {e}")
        return None