"""Translation into Russian with DeepL.

DeepL is given the surrounding conversation and the video's title as context,
so idioms and references come out right. Context characters don't count
against the monthly allowance.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

KEY_FILE = Path.home() / ".config" / "vtranslate" / "deepl-key"
SIGN_UP = "https://www.deepl.com/pro-api"
CHUNK = 25  # lines per request
CONTEXT_BEFORE = 1500  # characters of earlier conversation sent as context
CONTEXT_AFTER = 600  # characters of what comes next
# Whisper language codes that DeepL spells differently.
CODES = {"no": "NB", "nn": "NB"}
LANGUAGE_CODE = re.compile(r"^[a-z]{2,3}$")


class TranslationError(RuntimeError):
    """DeepL can't be used right now; the message says why and what to do."""


def key() -> str | None:
    found = os.environ.get("DEEPL_AUTH_KEY", "").strip()
    if not found and KEY_FILE.exists():
        found = KEY_FILE.read_text(encoding="utf-8").strip()
    return found or None


def save_key(value: str) -> Path:
    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    KEY_FILE.touch(mode=0o600, exist_ok=True)
    KEY_FILE.chmod(0o600)  # readable by this user only
    KEY_FILE.write_text(value.strip() + "\n", encoding="utf-8")
    return KEY_FILE


def _client(value: str | None = None):
    import deepl

    value = value or key()
    if not value:
        raise TranslationError("no DeepL key is set up; run: vtranslate --set-deepl-key")
    return deepl.DeepLClient(value, max_retries=5, timeout_s=60)


def _friendly(error: Exception) -> TranslationError:
    import deepl

    if isinstance(error, deepl.AuthorizationException):
        return TranslationError("DeepL rejected the key; set a new one with: vtranslate --set-deepl-key")
    if isinstance(error, deepl.QuotaExceededException):
        return TranslationError("this month's DeepL allowance is used up; it refills at the start of "
                                "your next monthly period, and a rerun then continues where it stopped")
    if isinstance(error, deepl.ConnectionException):
        return TranslationError("could not reach DeepL; check the internet connection and rerun "
                                "to continue where it stopped")
    return TranslationError(f"DeepL failed ({error})")


def check_ready(value: str | None = None) -> tuple[int, int]:
    """Make sure DeepL can be used before any slow work. Returns (used, limit)."""
    import deepl

    try:
        usage = _client(value).get_usage().character
    except deepl.DeepLException as error:
        raise _friendly(error) from error
    if usage.valid and usage.limit and usage.count >= usage.limit:
        raise _friendly(deepl.QuotaExceededException("quota exceeded"))
    return usage.count, usage.limit or 0


def check_language(language: str) -> str:
    """The DeepL code for a Whisper language code, or TranslationError."""
    import deepl

    code = CODES.get(language, language.upper())
    try:
        supported = {lang.code.upper() for lang in _client().get_source_languages()}
    except deepl.DeepLException as error:
        raise _friendly(error) from error
    if code not in supported:
        raise TranslationError(f"DeepL doesn't translate from this language ('{language}')")
    return code


def context_for(texts: list[str], start: int, end: int, title: str) -> str:
    """What comes before and after a chunk, which DeepL reads but doesn't translate."""
    before, after, size = [], [], 0
    for text in reversed(texts[:start]):
        size += len(text) + 1
        if size > CONTEXT_BEFORE:
            break
        before.insert(0, text)
    size = 0
    for text in texts[end:]:
        size += len(text) + 1
        if size > CONTEXT_AFTER:
            break
        after.append(text)
    parts = [f"Video title: {title}." if title else "",
             "Spoken transcript of a video, translated for a Russian voice-over."]
    if before:
        parts.append("Earlier: " + " ".join(before))
    if after:
        parts.append("Next: " + " ".join(after))
    return "\n".join(p for p in parts if p)


def translate(texts: list[str], language: str, title: str = "") -> list[str]:
    """The Russian for each line, in order. Lines go in chunks, each with context."""
    import deepl

    source = check_language(language)
    client = _client()
    try:
        used, limit = check_ready()
        needed = sum(len(text) for text in texts)
        if limit and used + needed > limit:
            raise TranslationError(
                f"this video needs about {needed:,} characters, but only {limit - used:,} of "
                f"this month's {limit:,} are left; it refills at the start of your next "
                "monthly period, and a rerun then continues where it stopped")
        results: list[str] = []
        for start in range(0, len(texts), CHUNK):
            end = min(start + CHUNK, len(texts))
            out = client.translate_text(
                texts[start:end], source_lang=source, target_lang="RU",
                context=context_for(texts, start, end, title),
                model_type="prefer_quality_optimized")
            results += [r.text.strip() for r in out]
            print(f"\r  translating: {end}/{len(texts)}", end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
    except deepl.DeepLException as error:
        raise _friendly(error) from error
    return results


def usage_note() -> str:
    try:
        used, limit = check_ready()
    except TranslationError:
        return ""
    return f"DeepL: {used:,} of {limit:,} characters used this month" if limit else ""
