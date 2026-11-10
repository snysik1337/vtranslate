"""Speech recognition, speaker labelling and sentence building."""
from __future__ import annotations

import bisect
import glob
import itertools
import os
import re
import shutil
import sys
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REINSTALL = ("curl -LsSf https://raw.githubusercontent.com/snysik1337/vtranslate/main/install.sh"
             " | sh")
MODELS = Path.home() / ".cache" / "vtranslate" / "models"
EMBEDDING_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
)
THREADS = max(1, (os.cpu_count() or 2) // 2)
WHISPER_BATCH = 8
RATE = 16000
STALE_DOWNLOAD = 3600  # seconds

SENTENCE_END = re.compile(r'[.!?…。！？؟]["»”\')\]]*$')
CLAUSE_END = re.compile(r"[,;:—–、，；：]$")
MAX_SENTENCE = 12.0  # seconds before splitting at a comma
HARD_MAX_SENTENCE = 20.0
PAUSE_SPLIT = 1.0
MERGE_GAP = 1.5  # short lines closer than this are spoken together
MERGE_MAX = 8.0
MIN_EMBED_SECONDS = 1.0  # shorter lines are matched to speakers found from longer ones
# Clusters that sound alike are one person; a cluster with too little speech to
# judge joins the closest voice. Measured on five real recordings: one person
# split into two clusters scored 0.88, 6-15 s samples of a person scored 0.75+
# against their own voice, and different people scored 0.13-0.48 (above 0.65 in
# under 1% of sampled cases).
MIN_OWN_VOICE_SECONDS = 6.0
SAME_PERSON = 0.65
MIN_SKIPPED_SPEECH = 2.5  # seconds of talking with no words before it is read again
MIN_RECOVERED_WORDS = 3
MAX_WORDS_PER_SECOND = 5
MIN_DIFFERENT_WORDS = 0.6  # share of words that must differ; loops repeat themselves
TOKENS_PER_SECOND = 12  # stops looping text quickly; more than any real speech needs
WHISPER_WINDOW = 30.0  # seconds Whisper decodes at a time
MAX_RECOVERIES = 40
MALE_FEMALE_F0 = 165.0  # Hz


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: int = 0


def cached_first(load, *args, **options):
    """Skip the online update check when a model is already on disk.

    A model folder can exist but be incomplete, for example when only one file
    of it was ever fetched. Opening it then fails with RuntimeError, and the
    online download fills in what is missing.
    """
    try:
        return load(*args, local_files_only=True, **options)
    except (OSError, RuntimeError):
        return load(*args, **options)


def transcribe(samples: np.ndarray, model_name: str, language: str | None) -> tuple[list[Word], str]:
    # The model hub library suggests an access token on first download; none is needed.
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    model = cached_first(WhisperModel, model_name, device="cpu", compute_type="int8",
                         cpu_threads=THREADS)
    segments, info = BatchedInferencePipeline(model).transcribe(
        samples,
        batch_size=WHISPER_BATCH,
        language=language,
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    print(f"  language: {info.language} ({info.language_probability:.0%})")
    words: list[Word] = []
    for seg in segments:
        words += [Word(w.start, w.end, w.word) for w in seg.words if w.word.strip()]
        _progress("recognising", seg.end / max(info.duration, 1))
    _progress_done()
    return _recover_skipped(model, samples, words, info.language), info.language


def find_gaps(speech: list[tuple[float, float]], words: list[Word]) -> list[tuple[float, float]]:
    """Stretches where someone is talking but no words were recognised."""
    gaps = []
    reach = list(itertools.accumulate((w.end for w in words), max))  # sorted, for bisect
    for start, end in speech:
        cursor = start
        # Only words reaching into this span count, so a gap between two spans is
        # not blamed on the later one. A binary search keeps long videos fast.
        index = bisect.bisect_right(reach, start)
        while index < len(words) and words[index].start < end:
            word = words[index]
            if word.start - cursor >= MIN_SKIPPED_SPEECH:
                gaps.append((cursor, word.start))
            cursor = max(cursor, word.end)
            index += 1
        if end - cursor >= MIN_SKIPPED_SPEECH:
            gaps.append((cursor, end))
    return gaps


def plausible(seg, words: list[str], seconds: float) -> bool:
    """Reject the invented or looping text Whisper can produce on unclear audio."""
    different = len({word.strip(" ,.!?").lower() for word in words})
    return (MIN_RECOVERED_WORDS <= len(words) <= MAX_WORDS_PER_SECOND * seconds
            and different >= MIN_DIFFERENT_WORDS * len(words)
            and seg.compression_ratio < 2.4
            and seg.avg_logprob > -1.0
            and seg.no_speech_prob < 0.6)


def _token_budget(seconds: float) -> int:
    """Whisper decodes 30 s at a time and the budget applies to each window, so a
    longer gap needs no more. Whisper also rejects budgets near its 448 limit."""
    return int(TOKENS_PER_SECOND * min(seconds, WHISPER_WINDOW)) + 8


def _recover_skipped(model, samples: np.ndarray, words: list[Word], language: str) -> list[Word]:
    """Read skipped speech again on its own.

    Whisper sometimes silently skips quiet speech, which would leave a hole in
    the dub. A separate speech detector shows where that happened. Only long
    gaps count, and only confident results, because short "gaps" are nearly
    always just imprecise word timing.
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    speech = [(s["start"] / RATE, s["end"] / RATE)
              for s in get_speech_timestamps(samples, VadOptions())]
    gaps = sorted(find_gaps(speech, words), key=lambda g: g[1] - g[0], reverse=True)
    recovered: list[Word] = []
    for start, end in gaps[:MAX_RECOVERIES]:
        offset = max(start - 0.2, 0.0)
        clip = samples[int(offset * RATE):int((end + 0.2) * RATE)]
        segments, _ = model.transcribe(clip, language=language, word_timestamps=True,
                                       vad_filter=False, condition_on_previous_text=False,
                                       temperature=0, max_new_tokens=_token_budget(end - start))
        for seg in segments:
            if plausible(seg, [w.word for w in seg.words], end - start):
                found = [Word(w.start + offset, w.end + offset, w.word) for w in seg.words]
                if not SENTENCE_END.search(found[-1].text):
                    found[-1].text += "."  # keep it from running into the next sentence
                recovered += found
    if recovered:
        print(f"  recovered {len(recovered)} skipped words")
    return sorted(words + recovered, key=lambda w: w.start)


def split_sentences(words: list[Word]) -> list[Segment]:
    """Group words into sentences, also breaking at long pauses."""
    sentences: list[Segment] = []
    current: list[Word] = []

    def flush() -> None:
        if current:
            sentences.append(Segment(current[0].start, current[-1].end,
                                     "".join(w.text for w in current).strip()))
            current.clear()

    for word in words:
        if current and (word.start - current[-1].end > PAUSE_SPLIT
                        or word.end - current[0].start > HARD_MAX_SENTENCE):
            flush()
        current.append(word)
        elapsed = word.end - current[0].start
        if SENTENCE_END.search(word.text) or (elapsed > MAX_SENTENCE and CLAUSE_END.search(word.text)):
            flush()
    flush()
    return sentences


def merge_short(segments: list[Segment]) -> list[Segment]:
    """Join quick fragments ("Yes. Exactly.") so the dub does not sound chopped."""
    merged: list[Segment] = []
    for seg in segments:
        last = merged[-1] if merged else None
        if (last and last.speaker == seg.speaker and seg.start - last.end < MERGE_GAP
                and seg.end - last.start < MERGE_MAX):
            last.end, last.text = seg.end, f"{last.text} {seg.text}"
        else:
            merged.append(Segment(**asdict(seg)))
    return merged


def _embedding_model() -> Path:
    model = MODELS / EMBEDDING_URL.rsplit("/", 1)[1]
    if not model.exists():
        print("  downloading speaker model")
        download_file(EMBEDDING_URL, model)
    return model


def download_file(url: str, dest: Path) -> None:
    """Fetch url to dest, refusing a short download so a cut-off file is never kept."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    # A run that was killed mid-download leaves its partial file behind. One
    # untouched for an hour belongs to no running download, so it is removed.
    for stale in dest.parent.glob(f"{glob.escape(dest.name)}.*.part"):
        try:
            if time.time() - stale.stat().st_mtime > STALE_DOWNLOAD:
                stale.unlink()
        except OSError:
            pass  # another run removed it first
    # A name of its own, so two runs fetching the same file never share it.
    fd, name = tempfile.mkstemp(dir=dest.parent, prefix=dest.name + ".", suffix=".part")
    os.close(fd)
    tmp = Path(name)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as out:
            expected = int(response.headers.get("Content-Length") or 0)
            shutil.copyfileobj(response, out)
        if expected and tmp.stat().st_size != expected:
            raise OSError(f"download of {dest.name} was cut short; run again to retry")
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)


def label_speakers(samples: np.ndarray, sentences: list[Segment], speakers: int | None,
                   threshold: float) -> None:
    """Work out who says each sentence.

    A voiceprint per sentence is about ten times cheaper than scanning the whole
    recording, and a sentence is the finest detail the dub can use anyway.
    """
    try:
        import sherpa_onnx as so
    except ImportError:
        print("  speaker detection is not installed, so everyone gets one voice; "
              f"to add it, run the install command again: {REINSTALL}")
        return

    # Measured: a few sentences at once, two threads each, is about 30% faster
    # than one sentence at a time on every thread, with the same voiceprints.
    # With few cores that means one sentence at a time on all of them, as before.
    together = max(1, THREADS // 2)
    extractor = so.SpeakerEmbeddingExtractor(so.SpeakerEmbeddingExtractorConfig(
        model=str(_embedding_model()), num_threads=max(1, THREADS // together)))

    def voiceprint(sentence: Segment) -> np.ndarray | None:
        stream = extractor.create_stream()
        stream.accept_waveform(RATE, samples[int(sentence.start * RATE):int(sentence.end * RATE)])
        stream.input_finished()
        if not extractor.is_ready(stream):
            return None
        return np.array(extractor.compute(stream), dtype=np.float32)

    voiceprints: list[np.ndarray | None] = []
    with ThreadPoolExecutor(together) as pool:
        for found in pool.map(voiceprint, sentences):  # in order, so progress only grows
            voiceprints.append(found)
            _progress("comparing voices", len(voiceprints) / len(sentences))
    _progress_done()
    lengths = [s.end - s.start for s in sentences]
    for sentence, label in zip(sentences, group_voices(voiceprints, lengths, speakers, threshold)):
        sentence.speaker = label


def _cluster(voices: np.ndarray, speakers: int | None, threshold: float) -> np.ndarray:
    import sherpa_onnx as so

    wanted = min(speakers, len(voices)) if speakers else -1
    config = so.FastClusteringConfig(num_clusters=wanted, threshold=threshold)
    return np.array(so.FastClustering(config)(voices))


def group_voices(voiceprints: list[np.ndarray | None], lengths: list[float],
                 speakers: int | None, threshold: float) -> list[int]:
    """Return a speaker number for each sentence, the biggest talker first."""
    usable = [i for i, v in enumerate(voiceprints)
              if v is not None and lengths[i] >= MIN_EMBED_SECONDS]
    if len(usable) < 2:
        return [0] * len(lengths)

    voices = np.stack([voiceprints[i] for i in usable])
    voices /= np.linalg.norm(voices, axis=1, keepdims=True)
    labels = _cluster(voices, speakers, threshold)

    talk = {int(c): sum(lengths[i] for i, l in zip(usable, labels) if l == c) for c in set(labels)}
    order = sorted(talk, key=talk.get, reverse=True)

    def centre(members: list[int]) -> np.ndarray:
        mean = voices[np.isin(labels, members)].mean(axis=0)
        return mean / np.linalg.norm(mean)

    people: list[list[int]] = []  # clusters grouped by person, biggest talker first
    for c in order:
        if speakers or not people:
            people.append([c])
            continue
        # One person can be split into several clusters (short lines sound
        # different from long ones), so a cluster that sounds like an earlier
        # person joins them. So does one with too little speech to judge.
        similarity = [centre([c]) @ centre(person) for person in people]
        best = int(np.argmax(similarity))
        if similarity[best] >= SAME_PERSON or talk[c] < MIN_OWN_VOICE_SECONDS:
            people[best].append(c)
        else:
            people.append([c])  # sounds like nobody else: a real, possibly brief, speaker
    centres = np.stack([centre(person) for person in people])

    result: list[int] = []
    for voiceprint in voiceprints:
        if voiceprint is None:  # too short to measure: assume the talker carries on
            result.append(result[-1] if result else 0)
        else:
            result.append(int(np.argmax(centres @ (voiceprint / np.linalg.norm(voiceprint)))))
    return result


def speaker_genders(samples: np.ndarray, segments: list[Segment]) -> dict[int, str]:
    """Guess male/female per speaker from the median pitch of up to 60 s of speech."""
    genders: dict[int, str] = {}
    for speaker in sorted({s.speaker for s in segments}):
        clips, total = [], 0.0
        for seg in (s for s in segments if s.speaker == speaker):
            clips.append(samples[int(seg.start * RATE):int(seg.end * RATE)])
            total += seg.end - seg.start
            if total > 60:
                break
        pitch = median_pitch(np.concatenate(clips))
        if pitch == 0.0:
            genders[speaker] = "male"  # unvoiced: no pitch to judge, default to male
            print(f"  speaker {speaker + 1}: unvoiced, defaulting to male")
        else:
            genders[speaker] = "female" if pitch > MALE_FEMALE_F0 else "male"
            print(f"  speaker {speaker + 1}: {pitch:.0f} Hz, {genders[speaker]}")
    return genders


def median_pitch(samples: np.ndarray) -> float:
    size, hop = 1024, 400
    if len(samples) < size:
        return 0.0
    frames = np.lib.stride_tricks.sliding_window_view(samples, size)[::hop] * np.hanning(size)
    frames = frames[np.abs(frames).mean(axis=1) > 0.003]
    if not len(frames):
        return 0.0
    spectrum = np.fft.rfft(frames, n=size * 2, axis=1)
    corr = np.fft.irfft(np.abs(spectrum) ** 2, axis=1)[:, :size]
    lo, hi = RATE // 350, RATE // 60
    lags = np.argmax(corr[:, lo:hi], axis=1) + lo
    strength = corr[np.arange(len(corr)), lags] / np.maximum(corr[:, 0], 1e-9)
    voiced = lags[strength > 0.5]
    return float(RATE / np.median(voiced)) if len(voiced) else 0.0


def _progress(label: str, fraction: float) -> None:
    print(f"\r  {label}: {min(fraction, 1):4.0%}", end="", file=sys.stderr, flush=True)


def _progress_done() -> None:
    print(file=sys.stderr)
