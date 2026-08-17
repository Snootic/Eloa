import argparse
import os
import subprocess
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from time import perf_counter
from typing import BinaryIO

import sounddevice as sd  # pyright:ignore[reportMissingTypeStubs]

from engine import AudioFormat, TTSEngine


@dataclass
class Performance:
    audio_seconds: float
    elapsed_seconds: float
    first_audio_seconds: float | None = None

    @property
    def realtime_speed(self) -> float:
        return self.audio_seconds / self.elapsed_seconds


def print_performance(name: str, result: Performance) -> None:
    print(f"\n{name}")
    if result.first_audio_seconds is not None:
        print(f"  Time to first audio: {result.first_audio_seconds:.2f}s")
    print(f"  Audio duration:      {result.audio_seconds:.2f}s")
    print(f"  Processing time:     {result.elapsed_seconds:.2f}s")
    print(f"  Generation speed:    {result.realtime_speed:.2f}x realtime")
    print(
        "  Result:              "
        + ("faster than realtime" if result.realtime_speed >= 1 else "below realtime")
    )


def benchmark_streaming_pcm16(
    engine: TTSEngine, text: str, sample_rate: int
) -> Performance:
    """Measure streaming synthesis without playback or audio encoding."""
    started_at = perf_counter()
    first_audio_at: float | None = None
    pcm_bytes = 0

    for chunk in engine.generate_stream(text, AudioFormat.PCM16):
        if first_audio_at is None:
            first_audio_at = perf_counter()
        pcm_bytes += len(chunk)

    finished_at = perf_counter()
    return Performance(
        audio_seconds=pcm_bytes / (sample_rate * 2),
        elapsed_seconds=finished_at - started_at,
        first_audio_seconds=(first_audio_at - started_at) if first_audio_at else None,
    )


def benchmark_pcm16_output(
    engine: TTSEngine, text: str, sample_rate: int
) -> Performance:
    """Measure the non-streaming generate_pcm16_output implementation."""
    started_at = perf_counter()
    audio = engine.generate_pcm16_output(text, AudioFormat.PCM16)
    finished_at = perf_counter()
    return Performance(
        audio_seconds=len(audio) / (sample_rate * 2),
        elapsed_seconds=finished_at - started_at,
    )


def run_benchmark(engine: TTSEngine, text: str, sample_rate: int) -> None:
    streaming = benchmark_streaming_pcm16(engine, text, sample_rate)
    non_streaming = benchmark_pcm16_output(engine, text, sample_rate)

    print_performance("Streaming PCM16", streaming)
    print_performance("Non-streaming PCM16", non_streaming)
    print("\nComparison")
    print(
        "  Streaming first-audio advantage: "
        + f"{non_streaming.elapsed_seconds - (streaming.first_audio_seconds or 0):.2f}s"
    )
    print(
        "  Total throughput winner:          "
        + (
            "streaming"
            if streaming.realtime_speed >= non_streaming.realtime_speed
            else "non-streaming"
        )
    )


def monitor_pcm(
    chunks: Iterator[bytes], sample_rate: int, bytes_per_sample: int = 2
) -> Iterator[bytes]:
    """Print synthesis throughput after forwarding a PCM stream."""
    started_at = perf_counter()
    first_audio_at: float | None = None
    pcm_bytes = 0

    try:
        for chunk in chunks:
            if first_audio_at is None:
                first_audio_at = perf_counter()
            pcm_bytes += len(chunk)
            yield chunk
    finally:
        finished_at = perf_counter()
        elapsed = finished_at - started_at
        audio_seconds = pcm_bytes / (sample_rate * bytes_per_sample)
        speed = audio_seconds / elapsed if elapsed else float("inf")

        print("\nPerformance")
        print(
            f"  Time to first audio: {(first_audio_at - started_at) if first_audio_at else 0:.2f}s"
        )
        print(f"  Audio duration:      {audio_seconds:.2f}s")
        print(f"  Processing time:     {elapsed:.2f}s")
        print(f"  Generation speed:    {speed:.2f}x realtime")
        if speed >= 1:
            print("  Result:              faster than realtime")
        else:
            print("  Result:              below realtime")


def decode_ogg_opus(opus_chunks: Iterator[bytes], sample_rate: int) -> Iterator[bytes]:
    """Decode streamed Ogg Opus bytes into PCM16 bytes for local playback."""
    decoder = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "ogg",
            "-i",
            "pipe:0",
            "-f",
            "s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    stdin = decoder.stdin
    stdout = decoder.stdout
    stderr = decoder.stderr
    assert stdin is not None
    assert stdout is not None
    assert stderr is not None

    def send_opus(output: BinaryIO) -> None:
        try:
            for chunk in opus_chunks:
                _ = output.write(chunk)
        finally:
            output.close()

    sender = threading.Thread(target=send_opus, args=(stdin,), daemon=True)
    sender.start()
    while pcm16_chunk := os.read(stdout.fileno(), 4096):
        yield pcm16_chunk

    sender.join()
    if decoder.wait() != 0:
        error = stderr.read().decode(errors="replace").strip()
        raise RuntimeError(f"Opus decoding failed: {error}")


def play_pcm(
    chunks: Iterator[bytes], sample_rate: int, dtype: str, bytes_per_sample: int
) -> None:
    buffered_chunks = []
    buffered_samples = 0
    minimum_buffer_samples = sample_rate // 2

    for chunk in chunks:
        buffered_chunks.append(chunk)
        buffered_samples += len(chunk) // bytes_per_sample
        if buffered_samples >= minimum_buffer_samples:
            break

    with sd.RawOutputStream(
        samplerate=sample_rate,
        channels=1,
        dtype=dtype,
        latency="high",
    ) as stream:
        for chunk in buffered_chunks:
            stream.write(chunk)
        for chunk in chunks:
            stream.write(chunk)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--benchmark",
    action="store_true",
    help="Measure generation only; do not play audio through the sound device.",
)
args = parser.parse_args()

model = TTSEngine("portuguese", "camilly-portuguese", None)

text = """
Me vejo no espelho, mas não me reconheço. Confesso que até de mim às vezes eu tenho medo. Nenhum sentimento, ou arrependimento, tudo se perdeu como folhas para o vento. Se sobrou algo de mim, no fundo lá dentro, luto para encontrar mas no fim eu nunca venço. E eu tento, mas é só perda de tempo, cada dia estou mais frio perdido nos pensamentos. Não preciso de razão, eu escolhi cair para dar de cara no chão. No meio da multidão, percebo a solidão, depois que você foi aqui virou escuridão. E quem vai entender? Se quero para mim morrer? Tanto faz, não importa mais nada e mais ninguém, não estou me importando nem comigo também...
"""

selected_format = AudioFormat.OPUS_24K

sample_rate = model.model.sample_rate

if args.benchmark:
    run_benchmark(model, text, sample_rate)
    raise SystemExit(0)

streamed_chunks = model.generate_stream(text, selected_format)

match selected_format:
    case AudioFormat.PCM16:
        pcm_chunks = streamed_chunks
    case AudioFormat.OPUS_24K | AudioFormat.OPUS_32K:
        pcm_chunks = decode_ogg_opus(streamed_chunks, sample_rate)
    case _:
        raise ValueError(f"Unsupported audio format: {selected_format}")

monitored_chunks = monitor_pcm(pcm_chunks, sample_rate)
play_pcm(monitored_chunks, sample_rate, "int16", 2)
