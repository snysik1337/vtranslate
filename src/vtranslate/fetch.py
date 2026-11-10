"""Turn a URL or local path into media files on disk.

For sites that keep sound and picture separate, the sound is fetched first and
the picture keeps downloading in the background, so work can start straight away.
"""
from __future__ import annotations

import copy
import hashlib
import re
import shutil
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

# YouTube slows each request after a fast start, so short requests finish three
# to four times quicker without opening extra connections. Only formats yt-dlp
# already downloads in pieces get this: other sites may not support it.
PIECE = 2 * 1024 * 1024


class SourceError(RuntimeError):
    """This source can never be dubbed, so a rerun would fail the same way."""


class Background:
    """A daemon thread, so Ctrl+C never hangs waiting for it."""

    def __init__(self, work) -> None:
        self.result = None
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._run, args=(work,), daemon=True)
        self.thread.start()

    def _run(self, work) -> None:
        try:
            self.result = work()
        except BaseException as error:  # re-raised on the main thread
            self.error = error

    @property
    def done(self) -> bool:
        return not self.thread.is_alive()

    def check(self) -> None:
        """Raise now if the work has already failed, rather than at the end."""
        if self.done and self.error:
            raise self.error

    def wait(self):
        self.thread.join()
        if self.error:
            raise self.error
        return self.result


@dataclass
class Source:
    audio: Path
    title: str
    video: Path | None = None
    pending: Background | None = None  # downloads the picture and returns its path
    stop: threading.Event | None = None  # set to stop that download

    def cancel(self) -> None:
        if self.stop:
            self.stop.set()

    def check(self) -> None:
        if self.pending:
            self.pending.check()

    def video_path(self) -> Path:
        """The picture file, waiting for its download if necessary."""
        if self.video is None:
            if not self.pending.done:
                print("  waiting for the video download to finish")
            self.video = self.pending.wait()
        return self.video


def is_url(value: str) -> bool:
    return re.match(r"^[a-z][a-z0-9+.-]*://", value, re.I) is not None


def job_id(source: str, quality: int) -> str:
    """Stable id, so an interrupted job resumes from the same cache folder.

    The quality is part of it: resuming a half-downloaded picture at another
    quality would splice two different streams into one corrupt file.
    """
    if is_url(source):
        key = f"{source}|{quality}"
    else:
        path = Path(source).expanduser().resolve()
        stat = path.stat()
        key = f"{path}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha1(key.encode("utf-8", "surrogateescape")).hexdigest()[:16]


def _deno_path() -> str | None:
    """yt-dlp needs a JS runtime for YouTube; ours is installed next to Python."""
    bundled = Path(sys.executable).parent / "deno"
    return str(bundled) if bundled.exists() else shutil.which("deno")


def _largest(folder: Path, stem: str) -> Path:
    files = [p for p in folder.glob(f"{stem}.*") if p.suffix not in {".part", ".ytdl", ".temp"}]
    if not files:
        raise RuntimeError("yt-dlp finished but produced no file")
    return max(files, key=lambda p: p.stat().st_size)


def download(url: str, work: Path, quality: int, cookies_from: str | None) -> Source:
    import yt_dlp

    class Downloader(yt_dlp.YoutubeDL):
        def dl(self, name, info, *args, **kwargs):
            options = info.get("downloader_options") or {}
            if options.get("http_chunk_size"):
                options = {**options, "http_chunk_size": min(options["http_chunk_size"], PIECE)}
                info = {**info, "downloader_options": options}
            return super().dl(name, info, *args, **kwargs)

    base = {
        # Prefer H.264/AAC: it copies into .mp4 and plays everywhere.
        "format_sort": [f"res:{quality}", "vcodec:h264", "acodec:m4a"],
        "noplaylist": True,
        "playlist_items": "1",  # a page holding several videos: use the first
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "concurrent_fragment_downloads": 4,
    }
    deno = _deno_path()
    if deno:
        base["js_runtimes"] = {"deno": {"path": deno}}
    if cookies_from:
        base["cookiesfrombrowser"] = (cookies_from,)

    stop = threading.Event()

    def stopped(progress: dict) -> None:
        if stop.is_set():
            raise yt_dlp.utils.DownloadCancelled("the dub stopped")

    found: dict = {}

    def fetch(stem: str, formats: str) -> tuple[dict, Path]:
        options = {**base, "format": formats, "outtmpl": str(work / f"{stem}.%(ext)s"),
                   "progress_hooks": [stopped]}
        try:
            with Downloader(options) as ydl:
                if found:  # reuse the first lookup: YouTube's takes about 5 seconds
                    info = ydl.process_ie_result(copy.deepcopy(found), download=True)
                else:
                    raw = ydl.extract_info(url, download=False, process=False)
                    if raw.get("_type", "video") == "video":
                        found.update(copy.deepcopy(raw))
                    info = ydl.process_ie_result(raw, download=True)
        except yt_dlp.utils.YoutubeDLError as error:
            # Often temporary (network, rate limits), so the job folder is kept.
            message = re.sub(r"^ERROR:\s*", "", str(error))
            if "ffmpeg" in message.lower() and not shutil.which("ffmpeg"):
                message += " (this site needs ffmpeg: install it with your package manager)"
            raise RuntimeError(message) from error
        if "entries" in info:
            entries = [e for e in info["entries"] or [] if e]
            if not entries:
                raise SourceError("no videos found at this address")
            if stem == "audio":
                print("  this address holds a list of videos; using the first")
            info = entries[0]
        done = (info.get("requested_downloads") or [{}])[0].get("filepath")
        return info, Path(done) if done else _largest(work, stem)

    info, audio = fetch("audio", "ba/b")
    title = info.get("title") or "video"
    if has_video(audio):  # sound and picture came as one file
        return Source(audio, title, video=audio)
    if all(f.get("vcodec") == "none" for f in info.get("formats") or [info]):
        raise SourceError("this address has sound but no video")

    def picture() -> Path:
        path = fetch("video", "bv*")[1]
        if not has_video(path):
            raise SourceError("this address has sound but no video")
        return path

    return Source(audio, title, pending=Background(picture), stop=stop)


def local(path: str) -> Source:
    media = Path(path).expanduser().resolve()
    if not has_video(media):
        raise SourceError("this file has no video")
    return Source(media, media.stem, video=media)


def _streams(media: Path):
    from .media import MediaError, inspect

    try:
        return inspect(media)
    except MediaError as error:
        raise SourceError(str(error)) from error


def has_audio(media: Path) -> bool:
    return _streams(media).channels > 0


def video_codec(media: Path) -> str:
    return _streams(media).codec


def has_video(media: Path) -> bool:
    return bool(video_codec(media))


def extract_speech_audio(media: Path, dest: Path) -> None:
    """Decode the first audio track, the one the final mix uses, to 16 kHz mono.

    It is written under a temporary name first, so an interrupted run never
    leaves a shortened file that a rerun would take as complete.
    """
    from .media import write_speech

    tmp = dest.with_name(dest.name + ".part")
    write_speech(media, tmp, 16000)
    tmp.replace(dest)
