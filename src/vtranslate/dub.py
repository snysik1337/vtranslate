"""Speak translated sentences, time them to the video and mix the final file."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import media
from .media import stretch
from .speech import THREADS, Segment, cached_first

VOICES = {
    "edge": {"male": ["ru-RU-DmitryNeural"], "female": ["ru-RU-SvetlanaNeural"]},
    "piper": {"male": ["dmitri", "denis", "ruslan"], "female": ["irina"]},
}
# When an engine runs out of voices, extra speakers differ by pitch.
PITCH_STEPS = [0, -14, 14, -26, 26, -36, 36]
EDGE_ATTEMPTS = 3
EDGE_PROBES = 2
CONCURRENCY = 4
GAP = 0.05
PHRASE_GAP = 1.0  # lines closer than this are spoken as one phrase
SYNC_LAG = 0.5  # how late a line may start relative to the original
SYNC_LEAD = 0.3  # how early a line may start, when that avoids speeding up
MAX_STRETCH = 1.15  # beyond this a line is spoken again instead of stretched
MAX_SILENT_SHARE = 0.1  # lines a voice may fail to speak before it counts as broken
ENVELOPE_RATE = 100
# Stream copy into .mp4 only works for these; anything else stays in .mkv.
MP4_CODECS = {"h264", "hevc", "av1", "vp9", "mpeg4"}


class VoiceServiceError(RuntimeError):
    """The online voice service would not produce a line of speech."""


@dataclass(frozen=True)
class Voice:
    name: str
    pitch: int = 0


@dataclass(frozen=True)
class Line:
    text: str
    voice: Voice
    speed: float = 1.0


@dataclass
class Placement:
    start: float
    speed: float  # how much faster than natural speech
    length: float  # final seconds of speech


def assign_voices(segments: list[Segment], genders: dict[int, str], forced: str | None,
                  engine: str) -> dict[int, Voice]:
    if forced:
        return {s.speaker: Voice(VOICES[engine][forced][0]) for s in segments}
    talk: dict[int, float] = {}
    for seg in segments:
        talk[seg.speaker] = talk.get(seg.speaker, 0.0) + seg.end - seg.start
    used = {"male": 0, "female": 0}
    voices = {}
    for speaker in sorted(talk, key=talk.get, reverse=True):
        gender = genders.get(speaker, "male")
        names = VOICES[engine][gender]
        turn = used[gender]
        used[gender] += 1
        pitch = PITCH_STEPS[(turn // len(names)) % len(PITCH_STEPS)]
        voices[speaker] = Voice(names[turn % len(names)], pitch)
    return voices


def plan(segments: list[Segment], natural: list[float], total: float, max_speed: float) -> list[Placement]:
    """Choose a start and speed for every line, keeping the dub in sync.

    Lines separated by short pauses form a phrase, spoken at one steady speed:
    the slowest that keeps every line within SYNC_LAG of its original start and
    ends the phrase before the next one begins. A little speed spread over a
    whole phrase sounds calmer than single rushed lines. A phrase may also start
    up to SYNC_LEAD early when that lowers the speed. When even max_speed can't
    keep up, the dub runs late and the next pause pulls it back in sync.
    """
    placements: list[Placement] = []
    cursor = 0.0
    first = 0
    while first < len(segments):
        last = first + 1
        while last < len(segments) and segments[last].start - segments[last - 1].end < PHRASE_GAP:
            last += 1
        limit = segments[last].start - GAP if last < len(segments) else total

        def place(speed: float, lead: float) -> tuple[list[Placement], bool]:
            out, at, in_sync = [], cursor, True
            for k in range(first, last):
                start = max(at, segments[k].start - lead)
                in_sync &= start <= segments[k].start + SYNC_LAG
                out.append(Placement(start, speed, natural[k] / speed))
                at = start + out[-1].length + GAP
            return out, in_sync and at - GAP <= limit + 1e-9

        speed, lead = _slowest(place, max_speed, 0.0), 0.0
        if speed > 1.0:
            early = _slowest(place, max_speed, SYNC_LEAD)
            if early < speed - 0.02:
                speed, lead = early, SYNC_LEAD
        phrase = place(speed, lead)[0]
        placements += phrase
        cursor = phrase[-1].start + phrase[-1].length + GAP
        first = last
    return placements


def _slowest(place, max_speed: float, lead: float) -> float:
    """The lowest speed at which a phrase fits, or max_speed if none does."""
    if place(1.0, lead)[1]:
        return 1.0
    if not place(max_speed, lead)[1]:
        return max_speed
    low, high = 1.0, max_speed
    for _ in range(24):  # faster always fits at least as well, so bisect
        middle = (low + high) / 2
        low, high = (low, middle) if place(middle, lead)[1] else (middle, high)
    return high


def has_words(text: str) -> bool:
    return re.search(r"\w", text) is not None


def trim(audio: np.ndarray, rate: int, threshold: float = 0.005) -> np.ndarray:
    """Remove the silence voice engines pad each line with."""
    loud = np.flatnonzero(np.abs(audio) > threshold)
    if not len(loud):
        return audio[:0]
    pad = rate // 50
    return audio[max(loud[0] - pad, 0):loud[-1] + pad]


def shift_pitch(audio: np.ndarray, factor: float) -> np.ndarray:
    """Resample so the voice sounds higher (factor > 1) or lower; speed moves with it."""
    return np.interp(np.arange(0, len(audio), factor), np.arange(len(audio)), audio).astype(np.float32)


class Counter:
    """Shows "speaking: 12/53" as lines finish; calling it passes a result through."""

    def __init__(self, total: int) -> None:
        self.done, self.total = 0, total

    def __call__(self, result=None):
        self.done += 1
        print(f"\r  speaking: {self.done}/{self.total}", end="", file=sys.stderr, flush=True)
        return result

    def finish(self) -> None:
        print(file=sys.stderr)


class Piper:
    """Offline Russian voices, generated straight into memory."""

    rate = 22050
    speaking_rates = {"dmitri": 18.0, "irina": 13.0, "denis": 15.0, "ruslan": 15.0}  # chars/s

    def __init__(self) -> None:
        self.voices = {}

    def chars_per_second(self, voice: Voice) -> float:
        return self.speaking_rates[voice.name]

    def _load(self, name: str) -> None:
        import onnxruntime
        from huggingface_hub import hf_hub_download
        from piper import PiperVoice
        from piper.config import PiperConfig

        base = f"ru/ru_RU/{name}/medium/ru_RU-{name}-medium.onnx"
        model = cached_first(hf_hub_download, "rhasspy/piper-voices", base)
        config = cached_first(hf_hub_download, "rhasspy/piper-voices", base + ".json")
        options = onnxruntime.SessionOptions()
        # Short lines parallelise best one per core, not many threads per line.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        # Loads a second faster than the default level and generates just as quickly.
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        # Its memory pool keeps growing on long videos; going without costs about 5% speed.
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        self.voices[name] = PiperVoice(
            session=onnxruntime.InferenceSession(
                model, sess_options=options, providers=["CPUExecutionProvider"]),
            config=PiperConfig.from_dict(json.loads(Path(config).read_text(encoding="utf-8"))),
        )

    def prepare(self, voices) -> None:
        names = {voice.name for voice in voices} - set(self.voices)
        if names:
            with ThreadPoolExecutor(len(names)) as pool:
                list(pool.map(self._load, names))

    def _speak(self, line: Line) -> np.ndarray:
        from piper import SynthesisConfig

        if not has_words(line.text):
            return np.zeros(0, dtype=np.float32)
        config = SynthesisConfig(length_scale=1 / line.speed)
        chunks = [c.audio_float_array
                  for c in self.voices[line.voice.name].synthesize(line.text, syn_config=config)]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(chunks)
        if line.voice.pitch:
            audio = shift_pitch(audio, 1 + line.voice.pitch / 200)
        return trim(audio, self.rate)

    def speak_many(self, lines: list[Line]) -> list[np.ndarray]:
        self.prepare(line.voice for line in lines)
        count = Counter(len(lines))
        with ThreadPoolExecutor(THREADS) as pool:
            result = list(pool.map(lambda line: count(self._speak(line)), lines))
        count.finish()
        return result


class Edge:
    """Microsoft's online voices. Lines are kept on disk so a rerun can reuse them."""

    rate = 24000

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        folder.mkdir(exist_ok=True)

    def chars_per_second(self, voice: Voice) -> float:
        return 14.0

    def _path(self, line: Line) -> Path:
        key = f"{line.text}|{line.voice.name}|{line.voice.pitch}|{line.speed:.2f}"
        return self.folder / f"{hashlib.sha1(key.encode()).hexdigest()}.mp3"

    async def _speak(self, line: Line, gate: asyncio.Semaphore) -> None:
        import edge_tts

        dest = self._path(line)
        if dest.exists():
            return
        if not has_words(line.text):
            dest.write_bytes(b"")
            return
        rate = f"{round((line.speed - 1) * 100):+d}%"
        pitch = f"{line.voice.pitch:+d}Hz"
        tmp = dest.with_suffix(".part")
        async with gate:
            for attempt in range(EDGE_ATTEMPTS):
                try:
                    await edge_tts.Communicate(line.text, line.voice.name, rate=rate,
                                               pitch=pitch).save(str(tmp))
                    if tmp.stat().st_size:
                        tmp.rename(dest)
                        return
                except Exception:
                    pass  # this free service drops requests often; retrying is normal
                await asyncio.sleep(0.5 * (attempt + 1))
        tmp.unlink(missing_ok=True)
        raise VoiceServiceError(f"no speech after {EDGE_ATTEMPTS} attempts: {line.text[:60]}")

    async def _speak_all(self, lines: list[Line]) -> None:
        gate = asyncio.Semaphore(CONCURRENCY)
        count = Counter(len(lines))

        async def run(line: Line) -> None:
            count(await self._speak(line, gate))

        await asyncio.gather(*(run(line) for line in lines))
        count.finish()

    def _decode(self, line: Line) -> np.ndarray:
        path = self._path(line)
        if path.stat().st_size == 0:
            return np.zeros(0, dtype=np.float32)
        return trim(media.read_mono(path, self.rate), self.rate)

    def speak_many(self, lines: list[Line]) -> list[np.ndarray]:
        # Identical lines share one file, so each is requested once. Requesting
        # them together would also make them overwrite each other's download.
        asyncio.run(self._speak_all(list(dict.fromkeys(lines))))
        with ThreadPoolExecutor(THREADS) as pool:
            return list(pool.map(self._decode, lines))

    def available(self, line: Line | None) -> bool:
        """Try one line first, so a failing service costs seconds and not a minute."""
        import edge_tts

        async def probe() -> bool:
            for _ in range(EDGE_PROBES):
                try:
                    async for chunk in edge_tts.Communicate(line.text[:80], line.voice.name).stream():
                        if chunk["type"] == "audio":
                            return True
                except Exception:
                    pass
            return False

        return line is None or asyncio.run(probe())


def first_line(segments: list[Segment], voices: dict[int, Voice]) -> Line | None:
    return next((Line(s.text, voices[s.speaker]) for s in segments if has_words(s.text)), None)


def synthesize(segments: list[Segment], voices: dict[int, Voice], total: float, max_speed: float,
               engine) -> tuple[np.ndarray, list[Placement]]:
    """Return the full-length dub track and where each line landed.

    Generating every line twice (once to measure, once at the right speed) is
    wasteful, so the first pass already uses a speed guessed from the text length.
    Small errors in the guess are stretched away; only bad guesses are spoken again.
    """
    guessed = [len(s.text) / engine.chars_per_second(voices[s.speaker]) for s in segments]
    speeds = [round(p.speed, 2) for p in plan(segments, guessed, total, max_speed)]
    lines = [Line(s.text, voices[s.speaker], speed) for s, speed in zip(segments, speeds)]
    clips = engine.speak_many(lines)

    natural = [len(clip) / engine.rate * speed for clip, speed in zip(clips, speeds)]
    placements = plan(segments, natural, total, max_speed)

    def tempo(i: int) -> float:
        return placements[i].speed / speeds[i]

    wrong = [i for i in range(len(clips)) if len(clips[i])
             and not 1 / MAX_STRETCH <= tempo(i) <= MAX_STRETCH]
    if wrong:
        fixed = [Line(segments[i].text, voices[segments[i].speaker], placements[i].speed)
                 for i in wrong]
        for i, clip in zip(wrong, engine.speak_many(fixed)):
            clips[i], speeds[i] = clip, placements[i].speed

    check_spoken(segments, clips)

    def fit(i: int) -> np.ndarray:
        if len(clips[i]) and len(clips[i]) / engine.rate > placements[i].length * 1.02:
            return stretch(clips[i], len(clips[i]) / engine.rate / placements[i].length, engine.rate)
        return clips[i]

    with ThreadPoolExecutor(THREADS) as pool:
        fitted = list(pool.map(fit, range(len(clips))))

    # Lines that run late can end after the video does; the track holds them all.
    ends = [p.start + len(clip) / engine.rate for clip, p in zip(fitted, placements)]
    track = np.zeros(int(max([total, *ends]) * engine.rate) + 1, dtype=np.float32)
    for clip, place in zip(fitted, placements):
        begin = int(place.start * engine.rate)
        track[begin:begin + len(clip)] += clip
    return track, placements


def check_spoken(segments: list[Segment], clips: list[np.ndarray]) -> None:
    """Refuse to produce a dub that is silently missing its speech.

    A voice can return no audio for a line that has words in it (Piper for text
    such as "_", Edge for a broken reply). A few such lines are reported; more
    than that fails the voice engine, so the caller can switch to another one.
    """
    wordy = [s.text for s in segments if has_words(s.text)]
    silent = [s.text for s, clip in zip(segments, clips) if has_words(s.text) and not len(clip)]
    if not silent:
        return
    if len(silent) > MAX_SILENT_SHARE * len(wordy):
        raise VoiceServiceError(f"no speech for {len(silent)} of {len(wordy)} lines, "
                                f"for example: {silent[0][:60]}")
    print(f"  could not speak {len(silent)} line(s), for example: {silent[0][:60]}")


def duck_envelope(segments: list[Segment], placements: list[Placement], total: float,
                  level: float) -> np.ndarray:
    """Gain curve for the original audio: lowered while anyone speaks."""
    size = int((total + 1) * ENVELOPE_RATE)
    busy = np.zeros(size, dtype=bool)
    spans = [(s.start, s.end) for s in segments]
    spans += [(p.start, p.start + p.length) for p in placements if p.length > 0]
    for start, end in spans:
        busy[int(start * ENVELOPE_RATE):int(end * ENVELOPE_RATE)] = True
    # Keep it low across short pauses so the original does not pop up mid-speech.
    # Padding with False on both sides makes every busy run a [start, end) pair,
    # even when speech starts at the very first moment of the video.
    edges = np.flatnonzero(np.diff(busy, prepend=False, append=False))
    for end, start in zip(edges[1:-1:2], edges[2::2]):
        if start - end < 1.5 * ENVELOPE_RATE:
            busy[end:start] = True
    gain = np.where(busy, level, 1.0).astype(np.float32)
    # 300 ms ramps; a centred box filter starts each dip just before the speech.
    # Repeating the end values keeps the very start and end from fading.
    width = int(0.3 * ENVELOPE_RATE)
    padded = np.pad(gain, (width // 2, width - 1 - width // 2), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid").astype(np.float32)


def match_loudness(track: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Bring the dub up to the loudness of the original speech."""
    def level(audio: np.ndarray) -> float:
        loud = audio[np.abs(audio) > 0.01]
        return float(np.sqrt(np.mean(loud ** 2))) if len(loud) else 0.0

    dub_level = level(track)
    if dub_level == 0:
        return track
    # Peaks above 1.0 are fine here: the final mix goes through a limiter.
    return track * np.clip(level(original) / dub_level, 0.5, 4.0)

