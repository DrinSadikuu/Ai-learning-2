
import asyncio
import base64

import sounddevice as sd
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
client = AsyncOpenAI()

SAMPLE_RATE = 24000
CHUNK_SIZE = 2400


async def send_microphone(connection):
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=CHUNK_SIZE
    ) as microphone:

        print("Microphone active. Start speaking!")

        while True:
            audio, overflowed = await asyncio.to_thread(
                microphone.read, CHUNK_SIZE
            )

            encoded_audio = base64.b64encode(audio).decode("utf-8")

            await connection.input_audio_buffer.append(
                audio=encoded_audio
            )


async def receive_audio(connection):
    with sd.RawOutputStream(
        samplerate=24000,
        channels=1,
        dtype="int16"
    ) as speaker:

        async for event in connection:

            if event.type == "input_audio_buffer.speech_started":
                print("\nYou are speaking...")

            elif event.type == "input_audio_buffer.speech_stopped":
                print("AI is thinking...")

            elif event.type == "response.output_audio.delta":
                audio_bytes = base64.b64decode(event.delta)
                await asyncio.to_thread(speaker.write, audio_bytes)

            elif event.type == "response.done":
                print("AI finished responding.\n")

            elif event.type == "error":
                print("Realtime API error:", event.error)


async def main():
    async with client.realtime.connect(
        model="gpt-realtime"
    ) as connection:

        await connection.session.update(
            session={
                "type": "realtime",
                "instructions": (
                    "You are a helpful AI voice assistant. "
                    "Speak naturally and keep your answers concise."
                ),
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": 24000
                        },
                        "turn_detection": {
                            "type": "server_vad"
                        }
                    },
                    "output": {
                        "format": {
                            "type": "audio/pcm",
                            "rate": 24000
                        },
                        "voice": "coral"
                    }
                }
            }
        )

        print("Connected to OpenAI Realtime API!")
        print("Speak into your microphone.")
        print("Press Ctrl+C to stop.\n")

        await asyncio.gather(
            send_microphone(connection),
            receive_audio(connection)
        )


try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("\nVoice assistant stopped.")
