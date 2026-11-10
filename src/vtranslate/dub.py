"""Speak translated sentences, time them to the video and mix the final file."""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import media, stress
from .media import stretch
from .speech import MODELS, REINSTALL, THREADS, Segment, download_file

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
# How late a line may start relative to the original. Measured on real dubs,
# 0.2 s costs only about 0.04x of pace against 0.5 s. A line never starts early:
# a voice that begins before the person on screen does is the easiest slip to see.
SYNC_LAG = 0.2
# How long after its speaker stops a line may still be spoken. Measured on real
# dubs, 0.3 s costs 0.02x to 0.05x of pace and ends every case of the voice
# talking on into a pause.
SYNC_OVERRUN = 0.3
LONGEST_STRETCH = 150  # lines of nonstop talk planned together
# The voices talk faster than most people, so when the timing leaves room they
# speak 8% slower than their natural pace; they speed up only when they must.
CALM_SPEED = 0.92
MAX_STEP = 0.15  # how much the speed may change from one line to the next in a phrase
LONGEST_PAUSE = 0.25  # seconds; longer pauses inside a spoken line are shortened
MAX_STRETCH = 1.15  # beyond this a line is spoken again instead of stretched
MAX_SILENT_SHARE = 0.1  # lines a voice may fail to speak before it counts as broken
ENVELOPE_RATE = 100
VOICE_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
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

    Lines separated by short pauses form a phrase. A phrase is spoken at the
    slowest steady speed that keeps every line within SYNC_LAG of its original
    start and ends before the next phrase begins, but never slower than
    CALM_SPEED. A little speed spread over several lines sounds calmer than
    single rushed lines. No line starts before its speaker does.

    Nonstop talk is one phrase many minutes long, and its tightest spot must
    not set the speed for all of it. So only the lines that make a spot tight
    get that speed; the lines before and after are planned again on their own,
    the same way, and come out slower.

    When even max_speed can't keep a phrase in sync, only the lines that cause
    the delay and those catching up after it are rushed.
    """
    # When each line must be finished: a little after its speaker stops.
    ends = [segment.end + SYNC_OVERRUN for segment in segments]
    placements: list[Placement] = []
    cursor = 0.0
    first = 0
    while first < len(segments):
        last = first + 1
        while last < len(segments) and segments[last].start - segments[last - 1].end < PHRASE_GAP:
            last += 1
        limit = segments[last].start - GAP if last < len(segments) else total
        # Hours of nonstop talk are planned a stretch at a time; planning them
        # whole takes time that grows with the square of their length. A
        # stretch may leave the next one its first line SYNC_LAG late.
        phrase: list[Placement] = []
        at, begin = cursor, first
        while begin < last:
            end = min(last, begin + LONGEST_STRETCH)
            stop = limit if end == last else segments[end].start + SYNC_LAG - GAP
            phrase += _stretch_of_talk(segments, natural, ends, begin, end, at, stop, max_speed)
            at, begin = phrase[-1].start + phrase[-1].length + GAP, end
        phrase = _smooth(segments, natural, first, phrase, cursor)
        placements += phrase
        cursor = phrase[-1].start + phrase[-1].length + GAP
        first = last
    return placements


def _stretch_of_talk(segments: list[Segment], natural: list[float], ends: list[float], first: int,
                     last: int, cursor: float, limit: float, max_speed: float) -> list[Placement]:
    """Plan lines first..last, which follow each other without a real pause."""
    if _fits(segments, natural, ends, first, last, cursor, limit, max_speed):
        return _steady(segments, natural, ends, first, last, cursor, limit, max_speed)
    rushed = _place(segments, natural, first, last, cursor, max_speed)
    out: list[Placement] = []
    for begin, end, late in _pieces(segments, ends, rushed, first, limit):
        if late:
            out += rushed[begin - first:end - first]
        else:
            at = out[-1].start + out[-1].length + GAP if out else cursor
            stop = segments[end].start - GAP if end < last else limit
            out += _steady(segments, natural, ends, begin, end, at, stop, max_speed)
    return out


def _smooth(segments: list[Segment], natural: list[float], first: int, phrase: list[Placement],
            cursor: float) -> list[Placement]:
    """Soften speed changes inside a phrase to at most MAX_STEP from line to line.

    Slower lines next to a faster one are sped up towards it, never the other
    way round: a faster line only ends sooner, so nothing falls out of sync.
    """
    speeds = [placed.speed for placed in phrase]
    for i in range(1, len(speeds)):
        speeds[i] = max(speeds[i], speeds[i - 1] - MAX_STEP)
    for i in range(len(speeds) - 2, -1, -1):
        speeds[i] = max(speeds[i], speeds[i + 1] - MAX_STEP)
    out, at = [], cursor
    for k, (placed, speed) in enumerate(zip(phrase, speeds), start=first):
        start = max(at, segments[k].start)
        out.append(Placement(start, speed, natural[k] / speed))
        at = start + out[-1].length + GAP
    return out


def _place(segments: list[Segment], natural: list[float], first: int, last: int, cursor: float,
           speed: float) -> list[Placement]:
    """Lines first..last one after another at one speed."""
    out, at = [], cursor
    for k in range(first, last):
        start = max(at, segments[k].start)
        out.append(Placement(start, speed, natural[k] / speed))
        at = start + out[-1].length + GAP
    return out


def _fits(segments: list[Segment], natural: list[float], ends: list[float], first: int, last: int,
          cursor: float, limit: float, speed: float) -> bool:
    """Whether _place at this speed keeps every line in sync and ends by the limit."""
    at = cursor
    for k in range(first, last):
        begin = segments[k].start
        start = max(at, begin)
        if start > begin + SYNC_LAG:
            return False
        at = start + natural[k] / speed + GAP
        if at - GAP > ends[k] + 1e-9:
            return False
    return at - GAP <= limit + 1e-9


def _steady(segments: list[Segment], natural: list[float], ends: list[float], first: int, last: int,
            cursor: float, limit: float, max_speed: float) -> list[Placement]:
    """Plan lines that fit at max_speed: steady speeds, each as slow as its spot allows."""
    out: list[Placement | None] = [None] * (last - first)
    todo = [(first, last, cursor, limit)]
    while todo:
        lo, hi, at, stop = todo.pop()

        def fits(speed: float) -> bool:
            return _fits(segments, natural, ends, lo, hi, at, stop, speed)

        speed = _slowest(fits, max_speed)
        placed = _place(segments, natural, lo, hi, at, speed)
        begin, end = (lo, hi) if speed <= CALM_SPEED else _tight(segments, ends, placed, lo, stop)
        out[begin - first:end - first] = placed[begin - lo:end - lo]
        if begin > lo:  # the lines before must leave the tight ones their start
            todo.append((lo, begin, at, placed[begin - lo].start - GAP))
        if end < hi:  # the lines after carry on where the tight ones stop
            before = placed[end - lo - 1]
            todo.append((end, hi, before.start + before.length + GAP, stop))
    return out


def _tight(segments: list[Segment], ends: list[float], placed: list[Placement], first: int,
           limit: float) -> tuple[int, int]:
    """The lines (begin, end) that set a run's speed: slowing them would break sync.

    The tightest spot is the line that starts closest to SYNC_LAG behind, the
    line that ends closest to its own deadline, or the end of the run when
    that is closer to its limit. The lines that push it there are the unbroken
    chain before it, back to the last line that started at its own time.
    """
    def pushed(k: int) -> bool:  # started late because the line before ran on
        return k > first and placed[k - first].start > segments[k].start + 1e-9

    last = first + len(placed)
    worst, end = placed[-1].start + placed[-1].length - limit, last
    for k in range(first, last):
        line = placed[k - first]
        if pushed(k) and line.start - segments[k].start - SYNC_LAG > worst:
            worst, end = line.start - segments[k].start - SYNC_LAG, k
        if line.start + line.length - ends[k] > worst:
            worst, end = line.start + line.length - ends[k], k + 1
    begin = end - 1
    while pushed(begin):
        begin -= 1
    return begin, end


def _pieces(segments: list[Segment], ends: list[float], rushed: list[Placement], first: int,
            limit: float) -> list[tuple[int, int, bool]]:
    """Split a phrase that is late even at top speed into (begin, end, late) runs.

    At top speed, a line that starts on its own time owes nothing to the lines
    before it, so the phrase can be cut there. A run is late when one of its
    lines starts more than SYNC_LAG behind or ends after its deadline, or when
    it overruns the phrase:
    nothing can help it, so it keeps top speed. The runs around it fit, and
    neighbouring ones are joined so they are planned together.
    """
    runs: list[list] = []
    for k, placed in enumerate(rushed, start=first):
        late = (placed.start > segments[k].start + SYNC_LAG
                or placed.start + placed.length > ends[k] + 1e-9)
        if runs and placed.start > segments[k].start + 1e-9:  # pushed back by the line before
            runs[-1][1], runs[-1][2] = k + 1, runs[-1][2] or late
        else:
            runs.append([k, k + 1, late])
    if rushed[-1].start + rushed[-1].length > limit + 1e-9:
        runs[-1][2] = True
    pieces: list[list] = []
    for begin, end, late in runs:
        if pieces and not late and not pieces[-1][2]:
            pieces[-1][1] = end
        else:
            pieces.append([begin, end, late])
    return [tuple(piece) for piece in pieces]


def _slowest(fits, max_speed: float) -> float:
    """The lowest speed at which fits(speed) holds, or max_speed if none does."""
    if fits(CALM_SPEED):
        return CALM_SPEED
    if not fits(max_speed):
        return max_speed
    low, high = CALM_SPEED, max_speed
    for _ in range(14):  # faster always fits at least as well, so bisect; ends within 0.0001
        middle = (low + high) / 2
        low, high = (low, middle) if fits(middle) else (middle, high)
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


def shorten_pauses(audio: np.ndarray, rate: int) -> np.ndarray:
    """Cut silences inside a line down to LONGEST_PAUSE, keeping their middle out.

    Some voices pause up to 0.7 s between sentences; that time is better spent
    speaking a little slower. "Silent" is relative to the line's loudest moment,
    so a quietly spoken line never loses soft speech.
    """
    if not len(audio):
        return audio
    threshold = min(0.005, 0.01 * float(np.abs(audio).max()))
    keep = int(LONGEST_PAUSE * rate)
    quiet = np.abs(audio) <= threshold
    edges = np.flatnonzero(np.diff(quiet.astype(np.int8), prepend=0, append=0))
    cuts = [(start + keep // 2, end - (keep - keep // 2))
            for start, end in zip(edges[::2], edges[1::2])
            if end - start > keep and start > 0 and end < len(audio)]  # inner pauses only
    if not cuts:
        return audio
    parts, at = [], 0
    for begin, finish in cuts:
        parts.append(audio[at:begin])
        at = finish
    parts.append(audio[at:])
    return np.concatenate(parts)


def shift_pitch(audio: np.ndarray, factor: float) -> np.ndarray:
    """Resample so the voice sounds higher (factor > 1) or lower; speed moves with it."""
    return np.interp(np.arange(0, len(audio), factor), np.arange(len(audio)), audio).astype(np.float32)


def parallel(work, items, threads: int = THREADS) -> list:
    """work(item) for every item, on several threads, results in order.

    On Ctrl+C or an error, map() cancels everything still queued, so only the
    few lines already running finish before it returns.
    """
    items = list(items)
    with ThreadPoolExecutor(max(1, min(threads, len(items)))) as pool:
        return list(pool.map(work, items))


class Counter:
    """Shows "speaking: 12/53" as lines finish; calling it passes a result through."""

    def __init__(self, total: int) -> None:
        self.done, self.total = 0, total
        self.lock = threading.Lock()

    def __call__(self, result=None):
        with self.lock:  # lines finish on several threads at once
            self.done += 1
            done = self.done
        print(f"\r  speaking: {done}/{self.total}", end="", file=sys.stderr, flush=True)
        return result

    def finish(self) -> None:
        print(file=sys.stderr)


class Piper:
    """Offline Russian voices, generated straight into memory."""

    rate = 22050
    # Characters a second at normal speed with long pauses shortened, measured
    # on the 72 lines of a real dub. Single lines differ from it by up to 10%.
    speaking_rates = {"dmitri": 19.0, "irina": 15.0, "denis": 16.5, "ruslan": 17.0}
    # Piper's length setting changes the speed far less than its number says:
    # 1/1.4 gives about 1.22x. Measured on all four voices, which agree within
    # 0.1x: the real speed for each setting. Asking by this table gives the
    # planner true lengths, so little is left to stretch afterwards.
    REAL_SPEED = [0.88, 0.955, 1.0, 1.045, 1.13, 1.26, 1.40, 1.50, 1.64]
    LENGTH_SCALE = [1.25, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]

    def __init__(self) -> None:
        self.voices = {}

    def chars_per_second(self, voice: Voice) -> float:
        # A pitch change is made by resampling, which changes the length with it.
        return self.speaking_rates[voice.name] * (1 + voice.pitch / 200)

    def length_scale(self, speed: float) -> float:
        """The Piper setting that really speaks this much faster than normal."""
        return float(np.interp(speed, self.REAL_SPEED, self.LENGTH_SCALE))

    def _load(self, name: str) -> None:
        import onnxruntime
        from piper import PiperVoice
        from piper.config import PiperConfig

        base = f"ru/ru_RU/{name}/medium/ru_RU-{name}-medium.onnx"

        model, config = MODELS / "piper" / base, MODELS / "piper" / (base + ".json")
        if not (model.exists() and config.exists()):
            print(f"  downloading {name} voice")
            for path in (model, config):
                if not path.exists():
                    download_file(VOICE_URL + str(path.relative_to(MODELS / "piper")), path)
        options = onnxruntime.SessionOptions()
        # Short lines parallelise best one per core, not many threads per line.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        # Loads a second faster than the default level and generates just as quickly.
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        # Its memory pool keeps growing on long videos; going without costs about 5% speed.
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        try:
            self.voices[name] = PiperVoice(
                session=onnxruntime.InferenceSession(
                    str(model), sess_options=options, providers=["CPUExecutionProvider"]),
                config=PiperConfig.from_dict(json.loads(config.read_text(encoding="utf-8"))),
            )
        except Exception as error:
            # A damaged file would fail on every run, so it is fetched again next
            # time. Other failures (memory, a library change) keep the download.
            damaged = isinstance(error, (ValueError, UnicodeDecodeError)) or any(
                sign in str(error) for sign in ("Protobuf", "INVALID_PROTOBUF"))
            if damaged:
                model.unlink(missing_ok=True)
                config.unlink(missing_ok=True)
                raise RuntimeError(f"the {name} voice file was damaged ({error}); "
                                   "run again to download it afresh") from error
            raise RuntimeError(f"the {name} voice could not be loaded ({error})") from error

    def prepare(self, voices) -> None:
        names = {voice.name for voice in voices} - set(self.voices)
        # Daemon threads, so Ctrl+C never waits for a voice download to finish.
        from .fetch import Background

        for loading in [Background(lambda name=name: self._load(name)) for name in names]:
            loading.wait()

    def _speak(self, line: Line) -> np.ndarray:
        from piper import SynthesisConfig

        if not has_words(line.text):
            return np.zeros(0, dtype=np.float32)
        config = SynthesisConfig(length_scale=self.length_scale(line.speed))
        chunks = [c.audio_float_array
                  for c in self.voices[line.voice.name].synthesize(stress.fix(line.text),
                                                                   syn_config=config)]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(chunks)
        if line.voice.pitch:
            audio = shift_pitch(audio, 1 + line.voice.pitch / 200)
        return trim(audio, self.rate)

    def speak_many(self, lines: list[Line]) -> list[np.ndarray]:
        self.prepare(line.voice for line in lines)
        count = Counter(len(lines))
        result = parallel(lambda line: count(self._speak(line)), lines)
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
        try:
            import edge_tts
        except ImportError as error:
            raise VoiceServiceError("online voices are not installed; to add them, run the "
                                    f"install command again: {REINSTALL}") from error

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
                tmp.unlink(missing_ok=True)  # never trust a leftover partial file
                try:
                    await edge_tts.Communicate(stress.fix(line.text, marks=False),
                                               line.voice.name, rate=rate,
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
            await self._speak(line, gate)
            count(None)

        tasks = [asyncio.create_task(run(line)) for line in lines]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            raise
        finally:
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
        return parallel(self._decode, lines)

    def available(self, line: Line | None) -> bool:
        """Try one line first, so a failing service costs seconds and not a minute."""
        try:
            import edge_tts
        except ImportError:
            return False

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
    clips = [shorten_pauses(clip, engine.rate) for clip in engine.speak_many(lines)]

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
            clips[i], speeds[i] = shorten_pauses(clip, engine.rate), placements[i].speed

    check_spoken(segments, clips)

    def fit(i: int) -> np.ndarray:
        # Stretch a line to its planned length. Measured on the clip itself: a
        # voice does not always speed up as much as it was asked to. Slowing
        # down stops at MAX_STRETCH; a line that short is left to end early.
        if not len(clips[i]):
            return clips[i]
        ratio = max(len(clips[i]) / engine.rate / placements[i].length, 1 / MAX_STRETCH)
        return stretch(clips[i], ratio, engine.rate) if abs(ratio - 1) > 0.02 else clips[i]

    fitted = parallel(fit, range(len(clips)))

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
    last = max([total] + [p.start + p.length for p in placements if p.length > 0])
    size = int((last + 1) * ENVELOPE_RATE)
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
    original_level = level(original)
    if original_level == 0:
        return track
    # Peaks above 1.0 are fine here: the final mix goes through a limiter.
    return track * np.clip(original_level / dub_level, 0.5, 4.0)

