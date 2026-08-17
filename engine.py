import os
import queue
import subprocess
import threading
from collections import OrderedDict
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path
from typing import BinaryIO, cast

import numpy as np
import torch
from pocket_tts import TTSModel, export_model_state
from torch import Tensor

import torching

type VoiceState = dict[str, dict[str, Tensor]]
type ModelKey = tuple[str, float, int, float | None, float, bool]

DEFAULT_VOICE_CONDITIONING = "hf://kyutai/tts-voices/alba-mackenna/casual.wav"


class AudioFormat(StrEnum):
    """Wire formats emitted by ``generate`` and ``generate_stream``."""

    PCM16 = "pcm16"
    OPUS_24K = "opus_24k"
    OPUS_32K = "opus_32k"


class TTSEngine:
    _voices: dict[str, VoiceState]
    __model__: TTSModel
    __current__voice__: VoiceState
    __ffmpeg__: subprocess.Popen[bytes] | None

    def __init__(
        self,
        language: str,
        voice_name: str | None,
        voice_path: str | None,
        max_cached_models: int = 2,
    ):
        if max_cached_models < 1:
            raise ValueError("max_cached_models must be at least 1")

        self._max_cached_models = max_cached_models
        self._models: OrderedDict[ModelKey, TTSModel] = OrderedDict()
        self._voice_caches: dict[ModelKey, dict[str, VoiceState]] = {}
        self._active_model_key: ModelKey | None = None
        self._state_lock = threading.RLock()
        self._voices = {}
        self.__model__ = self.load_model(language)

        name = voice_name or "alba"
        self.__current__voice__ = self.load_voice(
            voice_name=name,
            audio_conditioning=voice_path or DEFAULT_VOICE_CONDITIONING,
        )
        self.__ffmpeg__ = None

    def load_model(
        self,
        language: str,
        temp: float = 0.7,
        lsd_decode_steps: int = 2,
        noise_clamp: float | None = None,
        eos_threshold: float = -4.0,
        quantize: bool = False,
    ) -> TTSModel:
        key: ModelKey = (
            language,
            temp,
            lsd_decode_steps,
            noise_clamp,
            eos_threshold,
            quantize,
        )

        with self._state_lock:
            model = self._models.get(key)
            if model is None:
                model = TTSModel.load_model(
                    language=language,
                    temp=temp,
                    lsd_decode_steps=lsd_decode_steps,
                    noise_clamp=noise_clamp,
                    eos_threshold=eos_threshold,
                    quantize=quantize,
                )
                self._models[key] = model
                self._voice_caches[key] = {}

                if len(self._models) > self._max_cached_models:
                    evicted_key, _ = self._models.popitem(last=False)
                    del self._voice_caches[evicted_key]
            else:
                self._models.move_to_end(key)

            self._active_model_key = key
            self.__model__ = model
            self._voices = self._voice_caches[key]
            return model

    def load_voice(
        self,
        voice_name: str = "alba",
        audio_conditioning: Path | str | torch.Tensor | None = None,
    ) -> VoiceState:
        if voice_name not in self._voices:
            voice_path = Path("voices") / f"{voice_name}.safetensors"
            if voice_path.exists():
                source: Path | str | torch.Tensor = voice_path
            elif audio_conditioning is not None:
                source = audio_conditioning
            else:
                raise FileNotFoundError(
                    f"Voice '{voice_name}' was not found at {voice_path}"
                )

            self._voices[voice_name] = cast(
                VoiceState,
                self.__model__.get_state_for_audio_prompt(source),
            )

        self.__current__voice__ = self._voices[voice_name]
        return self.__current__voice__

    def generate_for(
        self,
        language: str,
        voice_name: str,
        text: str,
        audio_format: AudioFormat = AudioFormat.PCM16,
    ) -> Iterator[bytes]:
        """Select cached model and voice, then return a stream bound to that pair."""
        with self._state_lock:
            _ = self.load_model(language)
            _ = self.load_voice(voice_name)
            return self.generate_stream(text, audio_format)

    @staticmethod
    def _tensor_to_f32le(chunk: torch.Tensor) -> bytes:
        samples = (
            chunk.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
        )
        samples = np.clip(samples, -1.0, 1.0)
        return samples.astype("<f4", copy=False).tobytes()

    @staticmethod
    def _tensor_to_pcm(audio: Tensor, audio_format: AudioFormat) -> bytes:
        samples = audio.detach().cpu().numpy()
        samples = np.clip(samples, -1.0, 1.0)

        if audio_format is AudioFormat.PCM16:
            return (samples * 32767).astype(np.int16).tobytes()

        raise ValueError(f"{audio_format} requires the Opus encoder")

    def _generate_pcm(
        self,
        model: TTSModel,
        voice: VoiceState,
        text: str,
        pcm_format: AudioFormat = AudioFormat.PCM16,
    ) -> Iterator[bytes]:
        with torch.no_grad():
            for chunk in model.generate_audio_stream(  # pyright:ignore[reportUnknownMemberType]
                voice, text, frames_after_eos=5, copy_state=True
            ):
                yield self._tensor_to_pcm(chunk, pcm_format)

    def _generate_opus(
        self,
        model: TTSModel,
        voice: VoiceState,
        text: str,
        opus_format: AudioFormat = AudioFormat.OPUS_32K,
    ) -> Iterator[bytes]:
        process = self.spawn_ffmpeg_for_opus(opus_bitrate=opus_format)
        stdin = process.stdin
        stdout = process.stdout
        stderr = process.stderr
        assert stdin is not None
        assert stdout is not None
        assert stderr is not None
        writer_errors: queue.Queue[BaseException] = queue.Queue(maxsize=1)

        def write_audio(output: BinaryIO) -> None:
            try:
                with torch.no_grad():
                    for chunk in model.generate_audio_stream(
                        voice, text, frames_after_eos=5, copy_state=True
                    ):
                        _ = output.write(self._tensor_to_f32le(chunk))
            except Exception as error:
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
            self.__ffmpeg__ = None

        if not writer_errors.empty():
            raise writer_errors.get()
        if process.wait() != 0:
            error = stderr.read().decode(errors="replace").strip()
            raise RuntimeError(f"Opus encoding failed: {error}")

    def generate(
        self, text: str, audio_format: AudioFormat = AudioFormat.PCM16
    ) -> bytes:
        return b"".join(self.generate_stream(text, audio_format))

    def generate_pcm16_output(
        self, text: str, audio_format: AudioFormat = AudioFormat.PCM16
    ) -> bytes:
        """Test function for benchmarking"""
        with torch.no_grad():
            audio = self.__model__.generate_audio(
                text_to_generate=text,
                model_state=self.__current__voice__,
                frames_after_eos=5,
                copy_state=True,
            )
            return self._tensor_to_pcm(audio, audio_format)

    def generate_stream(
        self, text: str, audio_format: AudioFormat = AudioFormat.PCM16
    ) -> Iterator[bytes]:
        # Capture these now: a later request may select a different cached pair.
        model = self.__model__
        voice = self.__current__voice__
        if audio_format in (AudioFormat.PCM16):
            return self._generate_pcm(model, voice, text, audio_format)
        elif audio_format in (AudioFormat.OPUS_24K, AudioFormat.OPUS_32K):
            return self._generate_opus(model, voice, text, audio_format)
        else:
            raise ValueError(f"Unsupported audio format: {audio_format}")

    def clone_voice(self, sample_path: Path, voice_name: str) -> None:
        model_state = self.__model__.get_state_for_audio_prompt(sample_path)

        destination = Path(f"voices/{voice_name}.safetensors")
        export_model_state(model_state, destination)

    def spawn_ffmpeg_for_opus(
        self, opus_bitrate: AudioFormat
    ) -> subprocess.Popen[bytes]:
        if (opus_bitrate is not AudioFormat.OPUS_24K) and (
            opus_bitrate is not AudioFormat.OPUS_32K
        ):
            raise ValueError("Audio format not supported by opus writer")

        bitrate = "24k" if opus_bitrate is AudioFormat.OPUS_24K else "32k"

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
                str(self.__model__.sample_rate),
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
        self.__ffmpeg__ = process
        return process

    @property
    def model(self) -> TTSModel:
        return self.__model__

    @property
    def voices(self) -> dict[str, VoiceState]:
        return self._voices

    @property
    def ffmpeg(self) -> subprocess.Popen[bytes] | None:
        return self.__ffmpeg__
