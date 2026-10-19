"""FFmpeg-backed Opus encoding shared by every provider."""

import os
import queue
import subprocess
import threading
from collections.abc import Iterable, Iterator
from typing import BinaryIO

from tts_core.audio import AudioChunk, AudioFormat, to_f32le


def encode_opus(
    chunks: Iterable[AudioChunk], sample_rate: int, audio_format: AudioFormat
) -> Iterator[bytes]:
    """Encode float32 mono chunks as Ogg Opus, streaming through ffmpeg."""
    bitrate = audio_format.opus_bitrate
    if bitrate is None:
        raise ValueError(f"{audio_format} is not an Opus format")

    process = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            # Input: mono float32 PCM received through stdin
            "-f",
            "f32le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            # Output: voice-tuned Opus in an Ogg stream
            "-c:a",
            "libopus",
            "-application",
            "voip",
            "-b:a",
            bitrate,
            "-vbr",
            "constrained",
            "-frame_duration",
            "20",
            "-flush_packets",
            "1",
            "-f",
            "ogg",
            "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    stdin = process.stdin
    stdout = process.stdout
    stderr = process.stderr
    assert stdin is not None
    assert stdout is not None
    assert stderr is not None
    writer_errors: queue.Queue[BaseException] = queue.Queue(maxsize=1)

    def write_audio(output: BinaryIO) -> None:
        try:
            for chunk in chunks:
                _ = output.write(to_f32le(chunk))
        except Exception as error:  # noqa: BLE001  # re-raised in the consumer thread
            writer_errors.put(error)
        finally:
            output.close()

    writer = threading.Thread(target=write_audio, args=(stdin,), daemon=True)
    writer.start()
    try:
        while encoded_chunk := os.read(stdout.fileno(), 4096):
            yield encoded_chunk
    finally:
        if process.poll() is None:
            process.terminate()
        writer.join()

    if not writer_errors.empty():
        raise writer_errors.get()
    if process.wait() != 0:
        error = stderr.read().decode(errors="replace").strip()
        raise RuntimeError(f"Opus encoding failed: {error}")
