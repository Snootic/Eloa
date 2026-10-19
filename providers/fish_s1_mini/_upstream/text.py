"""Text cleanup and sentence splitting from fish-speech."""

import re
import string

SYMBOLS_MAPPING = {
    "\u2018": "'",
    "\u2019": "'",
}

REPLACE_SYMBOL_REGEX = re.compile(
    "|".join(re.escape(p) for p in SYMBOLS_MAPPING.keys())
)

EMOJI_REGEX = re.compile(
    "["
    "\U0001f600-\U0001f64f"
    "\U0001f300-\U0001f5ff"
    "\U0001f680-\U0001f6ff"
    "\U0001f1e0-\U0001f1ff"
    "]+",
    flags=re.UNICODE,
)


def clean_text(text: str) -> str:
    text = text.strip()
    text = REPLACE_SYMBOL_REGEX.sub(lambda match: SYMBOLS_MAPPING[match.group()], text)
    text = EMOJI_REGEX.sub("", text)
    return re.sub(r"[,]{2,}", lambda match: match.group()[0], text)


def _utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _break_text(texts, length: int, splits: set[str]):
    for text in texts:
        if _utf8_len(text) <= length:
            yield text
            continue
        current = ""
        for char in text:
            current += char
            if char in splits:
                yield current
                current = ""
        if current:
            yield current


def _break_text_by_length(texts, length: int):
    for text in texts:
        if _utf8_len(text) <= length:
            yield text
            continue
        current = ""
        for char in text:
            current += char
            if _utf8_len(current) >= length:
                yield current
                current = ""
        if current:
            yield current


def _add_cleaned(current: str, segments: list[str]) -> None:
    current = current.strip()
    if current and not all(c.isspace() or c in string.punctuation for c in current):
        segments.append(current)


def _protect_float(text: str) -> str:
    return re.sub(r"(\d+)\.(\d+)", r"<\1_f_\2>", text)


def _unprotect_float(text: str) -> str:
    return re.sub(r"<(\d+)_f_(\d+)>", r"\1.\2", text)


def split_text(text: str, length: int) -> list[str]:
    """Split text on sentence punctuation, then commas/spaces to fit length."""
    texts = [_protect_float(clean_text(text))]
    texts = _break_text(texts, length, {".", "!", "?", "\u3002", "\uff01", "\uff1f"})
    texts = map(_unprotect_float, texts)
    texts = _break_text(texts, length, {",", "\uff0c"})
    texts = _break_text(texts, length, {" "})
    texts = list(_break_text_by_length(texts, length))

    segments: list[str] = []
    current = ""
    for text in texts:
        if _utf8_len(current) + _utf8_len(text) <= length:
            current += text
        else:
            _add_cleaned(current, segments)
            current = text
    if current:
        _add_cleaned(current, segments)
    return segments
