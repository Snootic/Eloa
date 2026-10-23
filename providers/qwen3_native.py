"""ctypes binding for the qwentts.cpp shared C ABI.

The native handle owns the talker, code predictor, codec and GGML backend for
its lifetime. Synthesis streams float chunks through ``on_chunk``; the model is
kept loaded instead of spawning a fresh ``qwen-tts`` process per request.
"""

import ctypes
import queue
import threading
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from tts_core.audio import AudioChunk

CODE_BITS = 11
CODE_MASK = (1 << CODE_BITS) - 1
_END = object()

_FLOAT_PTR = ctypes.POINTER(ctypes.c_float)
_I32_PTR = ctypes.POINTER(ctypes.c_int32)
_CHUNK_CB = ctypes.CFUNCTYPE(ctypes.c_bool, _FLOAT_PTR, ctypes.c_int, ctypes.c_void_p)
_CANCEL_CB = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p)


class _QtAudio(ctypes.Structure):
    _fields_ = [
        ("samples", _FLOAT_PTR),
        ("n_samples", ctypes.c_int),
        ("sample_rate", ctypes.c_int),
        ("channels", ctypes.c_int),
    ]


class _QtInitParams(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_int),
        ("talker_path", ctypes.c_char_p),
        ("codec_path", ctypes.c_char_p),
        ("use_fa", ctypes.c_bool),
        ("clamp_fp16", ctypes.c_bool),
        ("max_batch", ctypes.c_int),
        ("codec_chunk_sec", ctypes.c_float),
        ("device", ctypes.c_char_p),
    ]


class _QtTtsParams(ctypes.Structure):
    _fields_ = [
        ("abi_version", ctypes.c_int),
        ("text", ctypes.c_char_p),
        ("lang", ctypes.c_char_p),
        ("instruct", ctypes.c_char_p),
        ("speaker", ctypes.c_char_p),
        ("ref_audio_24k", _FLOAT_PTR),
        ("ref_n_samples", ctypes.c_int),
        ("ref_text", ctypes.c_char_p),
        ("seed", ctypes.c_int64),
        ("max_new_tokens", ctypes.c_int),
        ("temperature", ctypes.c_float),
        ("top_k", ctypes.c_int),
        ("top_p", ctypes.c_float),
        ("repetition_penalty", ctypes.c_float),
        ("subtalker_temperature", ctypes.c_float),
        ("subtalker_top_k", ctypes.c_int),
        ("subtalker_top_p", ctypes.c_float),
        ("dump_dir", ctypes.c_char_p),
        ("cancel", _CANCEL_CB),
        ("cancel_user_data", ctypes.c_void_p),
        ("on_chunk", _CHUNK_CB),
        ("on_chunk_user_data", ctypes.c_void_p),
        ("ref_spk_emb", _FLOAT_PTR),
        ("ref_spk_dim", ctypes.c_int),
        ("ref_codes", _I32_PTR),
        ("ref_T", ctypes.c_int),
    ]


class QwenNativeBackend:
    """One resident qwentts.cpp context with streaming synthesis."""

    def __init__(
        self,
        library: str | Path,
        talker: str | Path,
        codec: str | Path,
        *,
        device: str = "Vulkan0",
        use_fa: bool = True,
        clamp_fp16: bool = False,
        max_batch: int = 1,
        codec_chunk_sec: float = 4.0,
    ):
        self._lib = ctypes.CDLL(str(library))
        self._configure_abi()

        iparams = _QtInitParams()
        self._lib.qt_init_default_params(ctypes.byref(iparams))
        self._talker = str(talker).encode()
        self._codec = str(codec).encode()
        self._device = device.encode() if device else None
        iparams.talker_path = self._talker
        iparams.codec_path = self._codec
        iparams.device = self._device
        iparams.use_fa = use_fa
        iparams.clamp_fp16 = clamp_fp16
        iparams.max_batch = max_batch
        iparams.codec_chunk_sec = codec_chunk_sec

        self._ctx = self._lib.qt_init(ctypes.byref(iparams))
        if not self._ctx:
            raise RuntimeError(self._error("qt_init"))
        self._codebooks = int(self._lib.qt_num_codebooks(self._ctx))
        if self._codebooks <= 0:
            self.close()
            raise RuntimeError("qwentts.cpp reported zero RVQ codebooks")
        self._voices: dict[
            tuple[str, str, int, int], tuple[np.ndarray, np.ndarray | None]
        ] = {}
        self._cancel_lock = threading.Lock()
        self._active_cancel: threading.Event | None = None

    def _configure_abi(self) -> None:
        lib = self._lib
        lib.qt_version.restype = ctypes.c_char_p
        lib.qt_last_error.restype = ctypes.c_char_p
        lib.qt_init_default_params.argtypes = [ctypes.POINTER(_QtInitParams)]
        lib.qt_init.argtypes = [ctypes.POINTER(_QtInitParams)]
        lib.qt_init.restype = ctypes.c_void_p
        lib.qt_free.argtypes = [ctypes.c_void_p]
        lib.qt_free.restype = None
        lib.qt_num_codebooks.argtypes = [ctypes.c_void_p]
        lib.qt_num_codebooks.restype = ctypes.c_int
        lib.qt_tts_default_params.argtypes = [ctypes.POINTER(_QtTtsParams)]
        lib.qt_synthesize.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_QtTtsParams),
            ctypes.POINTER(_QtAudio),
        ]
        lib.qt_synthesize.restype = ctypes.c_int
        lib.qt_audio_free.argtypes = [ctypes.POINTER(_QtAudio)]
        lib.qt_audio_free.restype = None

    def _error(self, where: str) -> str:
        raw = self._lib.qt_last_error()
        detail = raw.decode("utf-8", "replace") if raw else "unknown error"
        return f"qwentts.cpp {where} failed: {detail}"

    @property
    def version(self) -> str:
        raw = self._lib.qt_version()
        return raw.decode("utf-8", "replace") if raw else "unknown"

    def close(self) -> None:
        self.cancel()
        ctx = getattr(self, "_ctx", None)
        if ctx:
            self._lib.qt_free(ctx)
            self._ctx = None

    def cancel(self) -> None:
        """Ask the active qt_synthesize worker to stop."""
        with self._cancel_lock:
            event = self._active_cancel
        if event is not None:
            event.set()

    def __del__(self) -> None:
        self.close()

    def _voice_arrays(
        self, spk_path: Path, rvq_path: Path | None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        spk_stat = spk_path.stat()
        rvq_stat = rvq_path.stat() if rvq_path is not None else None
        key = (
            str(spk_path),
            str(rvq_path) if rvq_path else "",
            spk_stat.st_mtime_ns,
            rvq_stat.st_mtime_ns if rvq_stat else 0,
        )
        cached = self._voices.get(key)
        if cached is not None:
            return cached

        spk = np.fromfile(spk_path, dtype="<f4").astype(np.float32, copy=False)
        codes = None
        if rvq_path is not None:
            payload = rvq_path.read_bytes()
            count = (len(payload) * 8) // CODE_BITS
            if count == 0 or count % self._codebooks:
                raise ValueError(
                    f"Invalid RVQ file {rvq_path}: {count} codes for "
                    f"{self._codebooks} codebooks"
                )
            bits = int.from_bytes(payload, "little")
            flat = np.fromiter(
                ((bits >> (CODE_BITS * i)) & CODE_MASK for i in range(count)),
                dtype=np.int32,
                count=count,
            )
            codes = flat.reshape(self._codebooks, count // self._codebooks)
        result = (spk, codes)
        self._voices[key] = result
        return result

    def synthesize(
        self,
        text: str,
        spk_path: Path,
        rvq_path: Path | None,
        ref_text_path: Path | None,
        *,
        options: Mapping[str, Any],
    ) -> Iterator[AudioChunk]:
        if not self._ctx:
            raise RuntimeError("qwentts.cpp context is closed")
        spk, codes = self._voice_arrays(spk_path, rvq_path)

        text_bytes = text.encode("utf-8")
        language = str(options.get("language") or "").strip()
        lang_bytes = None if language.lower() in ("", "auto") else language.encode()
        ref_text_bytes = (
            ref_text_path.read_text(encoding="utf-8").strip().encode()
            if ref_text_path is not None
            else None
        )

        chunks: queue.Queue[np.ndarray | object] = queue.Queue()
        errors: list[str] = []
        cancel = threading.Event()
        with self._cancel_lock:
            self._active_cancel = cancel

        def on_chunk(ptr: _FLOAT_PTR, n_samples: int, _user: int) -> bool:
            if n_samples > 0:
                data = ctypes.string_at(ptr, n_samples * ctypes.sizeof(ctypes.c_float))
                chunks.put(np.frombuffer(data, dtype=np.float32).copy())
            return True

        def on_cancel(_user: int) -> bool:
            return cancel.is_set()

        chunk_cb = _CHUNK_CB(on_chunk)
        cancel_cb = _CANCEL_CB(on_cancel)

        params = _QtTtsParams()
        self._lib.qt_tts_default_params(ctypes.byref(params))
        params.text = text_bytes
        params.lang = lang_bytes
        params.ref_spk_emb = spk.ctypes.data_as(_FLOAT_PTR)
        params.ref_spk_dim = spk.size
        if codes is not None and ref_text_bytes:
            params.ref_codes = codes.ctypes.data_as(_I32_PTR)
            params.ref_T = codes.shape[1]
            params.ref_text = ref_text_bytes
        params.seed = int(options.get("seed", 42))
        if options.get("temperature") is not None:
            params.temperature = float(options["temperature"])
        if options.get("top_k") is not None:
            params.top_k = int(options["top_k"])
        if options.get("top_p") is not None:
            params.top_p = float(options["top_p"])
        if options.get("repetition_penalty") is not None:
            params.repetition_penalty = float(options["repetition_penalty"])
        if options.get("max_new_tokens") is not None:
            params.max_new_tokens = int(options["max_new_tokens"])
        params.on_chunk = chunk_cb
        params.cancel = cancel_cb

        out = _QtAudio()

        def work() -> None:
            try:
                rc = self._lib.qt_synthesize(
                    self._ctx, ctypes.byref(params), ctypes.byref(out)
                )
                if rc != 0:
                    errors.append(self._error(f"qt_synthesize ({rc})"))
            except Exception as error:  # noqa: BLE001  # surface callback-thread failures
                errors.append(str(error))
            finally:
                self._lib.qt_audio_free(ctypes.byref(out))
                chunks.put(_END)

        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        timeout = options.get("timeout")
        timeout_seconds = float(timeout) if timeout else None
        deadline = (
            time.monotonic() + timeout_seconds if timeout_seconds is not None else None
        )
        try:
            while True:
                try:
                    item = (
                        chunks.get(timeout=max(0.0, deadline - time.monotonic()))
                        if deadline is not None
                        else chunks.get()
                    )
                except queue.Empty:
                    errors.append(
                        f"Qwen3-TTS did not finish within {timeout_seconds:g}s"
                    )
                    break
                if item is _END:
                    break
                yield item  # type: ignore[misc]
        finally:
            cancel.set()
            worker.join()
            with self._cancel_lock:
                if self._active_cancel is cancel:
                    self._active_cancel = None
        if errors:
            raise RuntimeError(errors[0])
