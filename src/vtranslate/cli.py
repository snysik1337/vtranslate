"""vtranslate: download a video and dub it into Russian."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

CACHE = Path.home() / ".cache" / "vtranslate" / "jobs"
DEFAULT_OUTPUT = Path.home() / "Videos" / "vtranslate"
# Linux allows 255 bytes per file name; this leaves room for " [ru].tmp.mp4".
NAME_BYTES = 200


def cached(path: Path, compute, load=lambda data: data, dump=lambda value: value):
    """Save a slow step, so a rerun after a failure can skip it."""
    if path.exists():
        print("  (using saved result)")
        return load(json.loads(path.read_text(encoding="utf-8")))
    value = compute()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(dump(value), ensure_ascii=False), encoding="utf-8")
    tmp.rename(path)
    return value


class Stages:
    """Print each stage as it starts, and how long the previous one took."""

    def __init__(self, check) -> None:
        self.check = check
        self.mark = None

    def _lap(self) -> None:
        now = time.time()
        if self.mark is not None:
            print(f"  took {now - self.mark:.1f}s")
        self.mark = now

    def start(self, label: str) -> None:
        self.check()
        self._lap()
        print(label)

    def finish(self) -> None:
        self._lap()


def safe_name(title: str) -> str:
    name = re.sub(r'[/\\:*?"<>|\x00-\x1f]', " ", title)
    name = re.sub(r"\s+", " ", name).strip(" .")
    # Cut by bytes, not characters: a 100-character Japanese title is 300 bytes.
    name = name.encode("utf-8", "surrogateescape")[:NAME_BYTES].decode("utf-8", "ignore")
    return name.strip(" .") or "video"


def run(source: str, args: argparse.Namespace) -> Path:
    from . import fetch

    if not fetch.is_url(source) and not Path(source).expanduser().is_file():
        raise fetch.SourceError("not a URL or an existing file")
    work = CACHE / fetch.job_id(source, args.quality)
    work.mkdir(parents=True, exist_ok=True)
    try:
        output = _dub(source, args, work)
    except fetch.SourceError:
        shutil.rmtree(work, ignore_errors=True)  # a rerun cannot succeed, so keep nothing
        raise
    shutil.rmtree(work, ignore_errors=True)
    return output


def _dub(source: str, args: argparse.Namespace, work: Path) -> Path:
    from . import dub, fetch, media, speech
    from .fetch import SourceError
    from . import translate as deepl

    got = None
    stages = Stages(lambda: got and got.check())
    stages.start("[1/6] Getting the audio")
    if fetch.is_url(source):
        got = fetch.download(source, work, args.quality, args.cookies_from_browser)
    else:
        got = fetch.local(source)
    if not fetch.has_audio(got.audio):
        raise SourceError("this video has no audio track")
    speech_wav = work / "speech.wav"
    if not speech_wav.exists():
        fetch.extract_speech_audio(got.audio, speech_wav)
    samples = media.read_wav(speech_wav)
    total = len(samples) / speech.RATE
    print(f"  {got.title} ({total / 60:.1f} min)")

    stages.start(f"[2/6] Recognising speech (Whisper {args.model})")
    words, language = cached(
        work / f"words-{args.model}-{args.source_language or 'auto'}.json",
        lambda: speech.transcribe(samples, args.model, args.source_language),
        load=lambda d: ([speech.Word(**w) for w in d["words"]], d["language"]),
        dump=lambda v: {"words": [asdict(w) for w in v[0]], "language": v[1]},
    )
    if not words:
        raise SourceError("no speech found in this video")
    if language == "ru":
        raise SourceError("the speech is already in Russian")
    try:
        deepl.check_language(language)  # now, not after the speaker and voice work
    except deepl.TranslationError as error:
        if "doesn't translate" in str(error):
            raise SourceError(str(error)) from error
        raise

    multi = args.voice == "auto"
    stages.start("[3/6] Telling speakers apart" if multi
                 else "[3/6] Single voice, skipping speaker detection")
    sentences = speech.split_sentences(words)
    genders = {}
    if multi:
        speech.label_speakers(samples, sentences, args.speakers, args.speaker_threshold)
        genders = speech.speaker_genders(samples, sentences)
    segments = speech.merge_short(sentences)
    print(f"  {len(segments)} lines" + (f", {len(genders)} speaker(s)" if multi else ""))

    forced = None if multi else args.voice
    engines = ["edge", "piper"] if args.tts == "auto" else [args.tts]
    piper = dub.Piper()
    loading = None
    if engines[0] == "piper":  # get voices ready while translation runs
        wanted = dub.assign_voices(segments, genders, forced, "piper").values()
        loading = fetch.Background(lambda: piper.prepare(wanted))

    stages.start("[4/6] Translating into Russian")
    key = _text_key(segments, language, got.title)
    russian = cached(work / f"translated-{key}.json",
                     lambda: deepl.translate([s.text for s in segments], language, got.title))
    print_usage("  ", required=False)  # only a note: the translation is done and saved
    for seg, text in zip(segments, russian):
        seg.text = text

    stages.start("[5/6] Generating the Russian voice-over")
    if loading:
        loading.wait()
    for name in engines:
        voices = dub.assign_voices(segments, genders, forced, name)
        engine = piper if name == "piper" else dub.Edge(work / "tts")
        if name == "edge" and args.tts == "auto" and not engine.available(
                dub.first_line(segments, voices)):
            print("  online voices unavailable; using local voices")
            continue
        try:
            track, placements = dub.synthesize(segments, voices, total, args.max_speed, engine)
            break
        except dub.VoiceServiceError as error:
            if name == engines[-1]:
                raise
            # Switch the whole video, so voices never change halfway through.
            print(f"\n  online voices unavailable ({error}); using local voices")
    print(f"  voices: {name}")
    track = dub.match_loudness(track, samples)
    late = max(p.start - s.start for s, p in zip(segments, placements))
    print(f"  fastest line x{max(p.speed for p in placements):.2f}, latest line +{late:.1f}s")

    stages.start("[6/6] Mixing the final video")
    picture = got.video_path()
    envelope = dub.duck_envelope(segments, placements, total, args.original_volume)
    args.output.mkdir(parents=True, exist_ok=True)
    extension = ".mp4" if fetch.video_codec(picture) in dub.MP4_CODECS else ".mkv"
    output = args.output / f"{safe_name(got.title)} [ru]{extension}"
    tmp = output.with_name(f".{output.stem}.tmp{extension}")
    try:
        media.render(picture, got.audio, track, engine.rate, envelope, dub.ENVELOPE_RATE, tmp)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    tmp.replace(output)
    stages.finish()
    return output


def _text_key(segments, language: str, title: str) -> str:
    import hashlib

    joined = f"{language}\n{title}\n" + "\n".join(s.text for s in segments)
    return hashlib.sha1(joined.encode()).hexdigest()[:12]


def ask_for_key() -> bool:
    """Ask for a DeepL key in the terminal; True once a working key is saved.

    getpass keeps the key off the screen and out of the shell history.
    """
    import getpass

    from . import translate as deepl

    print("vtranslate translates with DeepL, which needs a free API key (one per person).")
    print(f"  1. Sign up for 'DeepL API Free' at {deepl.SIGN_UP}")
    print("  2. Copy the key from your account's API Keys page and paste it below.")
    print("It is saved for your user only, and you won't be asked again.")
    value = getpass.getpass("DeepL key (Enter to cancel): ").strip()
    if not value:
        return False
    try:
        used, limit = deepl.check_ready(value)
    except deepl.TranslationError as error:
        print(f"That key didn't work: {error}")
        return False
    print(f"Key works: {used:,} of {limit:,} characters used this month.")
    print(f"Saved to {deepl.save_key(value)}")
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from .translate import LANGUAGE_CODE

    parser = argparse.ArgumentParser(
        prog="vtranslate",
        description="Download a video (or use a local file) and dub it into Russian.",
    )
    parser.add_argument("sources", nargs="*", metavar="URL_OR_FILE")
    parser.add_argument("--set-deepl-key", action="store_true",
                        help="save a new DeepL API key (asked for in the terminal)")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT,
                        help="folder for finished videos (default: %(default)s)")
    parser.add_argument("-q", "--quality", type=int, default=1080,
                        help="maximum download height, e.g. 720 or 2160 (default: %(default)s)")
    parser.add_argument("--best", action="store_true",
                        help="use Microsoft's online voices when they work (nicer, slower)")
    parser.add_argument("-m", "--model", default="base",
                        choices=["tiny", "base", "small", "medium", "large-v3-turbo", "large-v3"],
                        help="Whisper speech model (default: base, the fastest accurate one; "
                             "small and large-v3-turbo are 3-7x slower but hear accents "
                             "and noise better)")
    parser.add_argument("--voice", default="auto", choices=["auto", "male", "female"],
                        help="auto: one voice per speaker; male/female: one voice for everything")
    parser.add_argument("--tts", choices=["auto", "edge", "piper"],
                        help="piper: offline voices (default); edge: online Microsoft voices; "
                             "auto: edge, switching to piper if it fails (default with --best)")
    parser.add_argument("--speakers", type=int,
                        help="exact number of speakers, if you know it (improves detection)")
    parser.add_argument("--speaker-threshold", type=float, default=0.75,
                        help="lower finds more speakers, higher finds fewer (default: %(default)s)")
    parser.add_argument("--from", dest="source_language", metavar="LANG",
                        help="spoken language code, e.g. en (default: detect automatically)")
    parser.add_argument("--original-volume", type=float, default=0.15,
                        help="volume of the original during speech, 0 to 1 (default: %(default)s)")
    parser.add_argument("--max-speed", type=float, default=1.6,
                        help="fastest allowed dub speech, at least 1 (default: %(default)s)")
    parser.add_argument("--cookies-from-browser", metavar="BROWSER",
                        help="use your browser's login for downloads, e.g. firefox")
    args = parser.parse_args(argv)
    if not args.sources and not args.set_deepl_key:
        parser.error("give at least one URL or file")

    # Check settings now, not after minutes of work.
    if args.source_language == "ru":
        parser.error("--from ru: the speech is already Russian")
    if args.source_language and not LANGUAGE_CODE.match(args.source_language):
        parser.error(f"--from {args.source_language}: use a language code such as en, de, es or zh")
    if args.quality <= 0:
        parser.error("--quality must be a positive height such as 720")
    if args.speakers is not None and args.speakers < 1:
        parser.error("--speakers must be at least 1")
    if args.speaker_threshold <= 0:
        parser.error("--speaker-threshold must be above 0")
    if not 0 <= args.original_volume <= 1:
        parser.error("--original-volume must be between 0 and 1")
    if args.max_speed < 1:
        parser.error("--max-speed must be at least 1")

    args.tts = args.tts or ("auto" if args.best else "piper")
    return args


def prepare_translation(args: argparse.Namespace) -> None:
    """Before any slow work: a DeepL key that works, allowance left, a known language."""
    from . import translate as deepl

    if args.set_deepl_key or not deepl.key():
        if not sys.stdin.isatty():
            raise SystemExit("error: no DeepL key is set up; run vtranslate --set-deepl-key "
                             "in a terminal first")
        if not ask_for_key():
            raise SystemExit("no key saved; vtranslate needs a DeepL key to translate")
        if not args.sources:
            raise SystemExit(0)
    try:
        print_usage()
        if args.source_language:
            deepl.check_language(args.source_language)
    except deepl.TranslationError as error:
        raise SystemExit(f"error: {error}") from error


def print_usage(indent: str = "", required: bool = True) -> None:
    """How much of the monthly DeepL allowance is used.

    When required, an unusable DeepL (no allowance left, offline) raises;
    otherwise the note is just left out.
    """
    from . import translate as deepl

    try:
        used, limit = deepl.check_ready()
    except deepl.TranslationError:
        if required:
            raise
        return
    if limit:
        print(f"{indent}DeepL: {used:,} of {limit:,} characters used this month")


def main() -> None:
    args = parse_args()
    prepare_translation(args)
    failed = 0
    for source in args.sources:
        began = time.time()
        try:
            output = run(source, args)
        except (RuntimeError, OSError) as error:
            # One bad source must not stop the rest of the batch.
            failed += 1
            print(f"error: {source}: {error}", file=sys.stderr)
            continue
        print(f"Done in {(time.time() - began) / 60:.1f} min: {output}")
    sys.exit(1 if failed else 0)
