"""Shared model/provider catalog used by providers and deployments."""

import json
import os
from pathlib import Path
from typing import Any

DEFAULT_CATALOG = Path("config/models.json")


def load_model_catalog(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``config/models.json`` (or ``TTS_MODEL_CATALOG``).

    The catalog is optional: providers fall back to their built-in aliases.
    """
    candidate = Path(path or os.getenv("TTS_MODEL_CATALOG", DEFAULT_CATALOG))
    if not candidate.is_file():
        return {}
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Could not read model catalog {candidate}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise TypeError(f"Model catalog {candidate} must be a JSON object")
    return value
