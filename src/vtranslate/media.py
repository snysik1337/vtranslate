"""Reading, filtering and writing sound and video with PyAV.

PyAV contains FFmpeg's own libraries (speech recognition needs it anyway), so
vtranslate needs no separate ffmpeg program: every step here runs the same
FFmpeg code the ffmpeg command would, inside this process.
"""
from __future__ import annotations

import functools
import wave
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

MIX_RATE = 48000
# Tags MP4 players know; Apple only plays HEVC tagged hvc1, so a copied picture
# keeps its tag when it is one of these, as the ffmpeg command did.
MP4_TAGS = {"avc1", "avc3", "hvc1", "hev1", "av01", "vp09", "mp4v"}


class MediaError(RuntimeError):
    """A file that FFmpeg cannot read or write."""


def picture_stream(container):
    """The video stream to dub. Cover art is stored as video but is not one."""
    for stream in container.streams.video:
        if not stream.disposition & av.stream.Disposition.attached_pic:
            return stream
    return None


@dataclass(frozen=True)
class Info:
    channels: int  # of the first audio track; 0 when there is no sound
    codec: str  # of the picture; "" when there is none (cover art does not count)
    tag: str


def inspect(path: Path) -> Info:
    """What a file contains. Each file is opened once, however often it is asked."""
    try:
        stat = path.stat()
        return _inspect(str(path), stat.st_size, stat.st_mtime_ns)
    except (OSError, av.error.FFmpegError) as error:
        raise MediaError(f"cannot read this file ({error})") from error


@functools.lru_cache(maxsize=32)
def _inspect(path: str, size: int, mtime: int) -> Info:
    with av.open(path) as container:
        audio = container.streams.audio
        picture = picture_stream(container)
        return Info(audio[0].codec_context.channels or 2 if audio else 0,
                    picture.codec_context.name if picture else "",
                    picture.codec_context.codec_tag if picture else "")


def _frames(container, stream):
    """Decoded frames, skipping damaged packets the way the ffmpeg command does."""
    for packet in container.demux(stream):
        try:
            yield from packet.decode()
        except av.error.InvalidDataError:
            continue


def decode(path: Path, rate: int, layout: str) -> Iterator[np.ndarray]:
    """The first audio track as float32 blocks shaped (channels, samples)."""
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise MediaError(f"{path.name} has no sound")
            stream = container.streams.audio[0]
            resampler = av.AudioResampler(format="fltp", layout=layout, rate=rate)
            for frame in _frames(container, stream):
                for out in resampler.resample(frame):
                    yield out.to_ndarray()
            for out in resampler.resample(None):
                yield out.to_ndarray()
    except av.error.FFmpegError as error:
        raise MediaError(f"cannot read {path.name} ({error})") from error


def read_mono(path: Path, rate: int) -> np.ndarray:
    blocks = [block[0] for block in decode(path, rate, "mono")]
    return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)


def write_speech(media: Path, dest: Path, rate: int) -> None:
    """The first audio track as a mono 16-bit WAV, written as it is decoded."""
    with wave.open(str(dest), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        for block in decode(media, rate, "mono"):
            pcm = np.clip(np.round(block[0] * 32768), -32768, 32767).astype("<i2")
            out.writeframes(pcm.tobytes())


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as source:
        pcm = source.readframes(source.getnframes())
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768


def _frame(block: np.ndarray, rate: int, layout: str, pts: int):
    frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(block, dtype=np.float32),
                                       format="fltp", layout=layout)
    frame.sample_rate = rate
    frame.time_base = Fraction(1, rate)
    frame.pts = pts
    return frame


def filtered(blocks: Iterable[np.ndarray], rate: int, layout: str, name: str, options: str):
    """Run blocks of sound through one FFmpeg filter, yielding the frames it gives back."""
    graph = av.filter.Graph()
    source = graph.add_abuffer(format="fltp", sample_rate=rate, layout=layout,
                               time_base=Fraction(1, rate))
    step = graph.add(name, options)
    sink = graph.add("abuffersink")
    source.link_to(step)
    step.link_to(sink)
    graph.configure()

    def ready():
        while True:
            try:
                yield sink.pull()
            except (av.error.BlockingIOError, av.error.EOFError):
                return

    pts = 0
    for block in blocks:
        if block.shape[1]:
            source.push(_frame(block, rate, layout, pts))
            pts += block.shape[1]
            yield from ready()
    source.push(None)
    yield from ready()


def stretch(audio: np.ndarray, tempo: float, rate: int) -> np.ndarray:
    """Change speed without changing pitch."""
    try:
        out = [frame.to_ndarray()[0]
               for frame in filtered([audio[None]], rate, "mono", "atempo", f"{tempo:.4f}")]
    except av.error.FFmpegError as error:
        raise MediaError(f"cannot change the speed of a line ({error})") from error
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def resampled(audio: np.ndarray, rate: int, to: int) -> Iterator[np.ndarray]:
    """Mono audio at another sample rate, a second at a time."""
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=to)
    for start in range(0, len(audio), rate):
        for out in resampler.resample(_frame(audio[None, start:start + rate], rate, "mono", start)):
            yield out.to_ndarray()
    for out in resampler.resample(None):
        yield out.to_ndarray()


class Reader:
    """Exactly n samples at a time from a sequence of differently sized blocks."""

    def __init__(self, blocks: Iterable[np.ndarray], channels: int) -> None:
        self.blocks = iter(blocks)
        self.channels = channels
        self.held = np.zeros((channels, 0), dtype=np.float32)
        self.ended = False

    @property
    def finished(self) -> bool:
        return self.ended and not self.held.shape[1]

    def read(self, n: int) -> tuple[np.ndarray, int]:
        """n samples, padded with silence after the end, and how many were real."""
        parts, have = [self.held], self.held.shape[1]
        while have < n and not self.ended:
            try:
                block = next(self.blocks)
            except StopIteration:
                self.ended = True
                break
            parts.append(block)
            have += block.shape[1]
        data = np.concatenate(parts, axis=1) if len(parts) > 1 else self.held
        out, self.held = data[:, :n], data[:, n:]
        real = out.shape[1]
        if real < n:
            out = np.pad(out, ((0, 0), (0, n - real)))
        return out, real


def mix(original: Path, voice: np.ndarray, rate: int, envelope: np.ndarray,
        envelope_rate: int) -> Iterator[np.ndarray]:
    """The original turned down by the envelope, with the voice track on top, in stereo.

    It runs as long as the longer of the two, so a line still being spoken as the
    video ends is kept. A mono original is copied to both sides at full level.
    """
    mono = inspect(original).channels == 1
    sides = decode(original, MIX_RATE, "mono" if mono else "stereo")
    if mono:
        sides = (np.repeat(block, 2, axis=0) for block in sides)
    background = Reader(sides, 2)
    speech = Reader(resampled(voice, rate, MIX_RATE), 1)
    position = 0
    while not (background.finished and speech.finished):
        quiet, heard = background.read(MIX_RATE)
        spoken, said = speech.read(MIX_RATE)
        times = (position + np.arange(MIX_RATE)) * (envelope_rate / MIX_RATE)
        gain = np.interp(times, np.arange(len(envelope)), envelope).astype(np.float32)
        length = max(heard, said)
        if length:
            yield (quiet * gain + spoken)[:, :length]
        position += MIX_RATE


def render(video: Path, audio: Path, voice: np.ndarray, rate: int, envelope: np.ndarray,
           envelope_rate: int, output: Path) -> None:
    """Copy the picture unchanged and add the mixed sound as AAC."""
    sound_frames = filtered(mix(audio, voice, rate, envelope, envelope_rate), MIX_RATE,
                            "stereo", "alimiter", "limit=0.89:level=disabled")
    options = {"movflags": "+faststart"} if output.suffix == ".mp4" else {}
    try:
        with av.open(str(video)) as source, av.open(str(output), "w", options=options) as out:
            picture = picture_stream(source)
            if picture is None:
                raise MediaError(f"{video.name} has no video")
            copy = out.add_stream_from_template(picture)
            tag = picture.codec_context.codec_tag
            if output.suffix == ".mp4" and tag in MP4_TAGS:
                copy.codec_context.codec_tag = tag
            sound = out.add_stream("aac", rate=MIX_RATE, layout="stereo",
                                   options={"aac_coder": "fast"})
            sound.bit_rate = 128_000
            sound.metadata["language"] = "rus"
            # Like the ffmpeg command, start the picture at zero even when the
            # file's timestamps do not, so it lines up with the sound.
            start = Fraction(source.start_time or 0, av.time_base)
            shift = int(start / picture.time_base) if picture.time_base else 0
            heard = 0  # samples of sound written so far

            def sound_until(seconds: float | None) -> None:
                nonlocal heard
                while seconds is None or heard < seconds * MIX_RATE:
                    frame = next(sound_frames, None)
                    if frame is None:
                        return
                    heard += frame.samples
                    out.mux(sound.encode(frame))

            # Interleave picture and sound by time, so players read the file evenly.
            for packet in source.demux(picture):
                if not packet.size:  # the empty packet that ends the stream
                    continue
                # Some files (Matroska with B-frames) leave out a few decode
                # times; the muxer works them out, so such packets are kept.
                stamp = packet.dts if packet.dts is not None else packet.pts
                if stamp is not None:
                    sound_until(float((stamp - shift) * picture.time_base) + 0.5)
                if packet.dts is not None:
                    packet.dts -= shift
                if packet.pts is not None:
                    packet.pts -= shift
                packet.stream = copy
                out.mux(packet)
            sound_until(None)
            out.mux(sound.encode(None))
    except av.error.FFmpegError as error:
        raise MediaError(f"cannot write the dubbed video ({error})") from error
