import argparse
import asyncio
import json
from pathlib import Path

import websockets


async def main() -> None:
    parser = argparse.ArgumentParser(description="Pocket TTS WebSocket test client")
    parser.add_argument("--url", default="ws://127.0.0.1:8082/v1/audio/speech/stream")
    parser.add_argument("--model", default="english")
    parser.add_argument("--voice", default="alba")
    parser.add_argument("--format", choices=("opus_24k", "opus_32k", "pcm16"), default="opus_24k")
    parser.add_argument("--output", type=Path)
    parser.add_argument("text", nargs="?", default="Hello from the Pocket TTS WebSocket client.")
    args = parser.parse_args()
    output_path = args.output or Path(
        "speech.pcm" if args.format == "pcm16" else "speech.ogg"
    )

    request = {
        "model": args.model,
        "voice": args.voice,
        "input": args.text,
        "response_format": args.format,
    }

    audio_bytes = 0
    async with websockets.connect(args.url) as websocket:
        await websocket.send(json.dumps(request))

        with output_path.open("wb") as output:
            async for message in websocket:
                if isinstance(message, bytes):
                    output.write(message)
                    audio_bytes += len(message)
                    continue

                event = json.loads(message)
                print(event)
                if event["type"] == "speech.done":
                    break
                if event["type"] == "error":
                    raise RuntimeError(event["message"])

    print(f"Saved {audio_bytes} bytes to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
