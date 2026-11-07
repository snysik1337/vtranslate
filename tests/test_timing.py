import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

from vtranslate import dub, speech
from vtranslate.dub import Line, Voice, assign_voices, plan
from vtranslate.speech import (Segment, Word, find_gaps, group_voices, merge_short,
                               split_sentences)


def test_sentences_end_at_punctuation_and_long_pauses():
    words = [Word(0, .4, " Hello", ), Word(.4, .8, " there."), Word(1, 1.3, " Next"),
             Word(4, 4.3, " after"), Word(4.3, 4.6, " pause.")]
    assert [s.text for s in split_sentences(words)] == ["Hello there.", "Next", "after pause."]


def test_words_join_without_inventing_spaces():
    words = [Word(0, .2, "你"), Word(.2, .4, "好。")]
    assert split_sentences(words)[0].text == "你好。"


def test_skipped_speech_is_found_but_imprecise_word_timing_is_ignored():
    words = [Word(4, 5, " a"), Word(6, 7, " b"), Word(11, 12, " c")]
    assert find_gaps([(0, 8)], words) == [(0, 4)]  # the 1 s and 1 s pauses are ignored
    assert find_gaps([(10, 11.5), (20, 23)], words) == [(20, 23)]  # no words at all there
    assert find_gaps([(0, 2)], []) == []


def test_recovered_text_must_be_plausible():
    from types import SimpleNamespace as Seg

    good = Seg(compression_ratio=1.1, avg_logprob=-0.7, no_speech_prob=0.2)
    words = "Welcome to American English with Sushil".split()
    assert speech.plausible(good, words, 3.7)
    assert not speech.plausible(good, words[:2], 3.7)  # a stray word or two
    assert not speech.plausible(good, words * 30, 2.9)  # far too many words for the time
    loop = "Well, it's not that bad, it's not that bad, it's not".split()
    assert not speech.plausible(good, loop, 3.0)  # short enough, but it repeats itself
    confused = Seg(compression_ratio=1, avg_logprob=-1.5, no_speech_prob=0.2)
    assert not speech.plausible(confused, words, 3.7)
    silent = Seg(compression_ratio=1, avg_logprob=-0.7, no_speech_prob=0.9)
    assert not speech.plausible(silent, words, 3.7)
    looped = Seg(compression_ratio=8, avg_logprob=-0.7, no_speech_prob=0.2)
    assert not speech.plausible(looped, words, 3.7)


def test_quick_lines_merge_only_within_one_speaker():
    lines = [Segment(0, 1, "Yes.", 0), Segment(1.2, 2, "Right.", 0), Segment(2.2, 3, "No.", 1)]
    assert [s.text for s in merge_short(lines)] == ["Yes. Right.", "No."]


def fake_clusters(monkeypatch, labels):
    monkeypatch.setattr(speech, "_cluster", lambda voices, speakers, threshold: np.array(labels))


def test_tiny_cluster_folds_into_the_closest_main_voice(monkeypatch):
    fake_clusters(monkeypatch, [0, 0, 1, 1, 2])
    a, b = np.array([1, 0, 0.]), np.array([0, 1, 0.])
    stray = np.array([0.2, 1, 0.])  # most like voice b
    result = group_voices([a, a, b, b, stray], [40, 40, 40, 40, 2], None, 0.75)
    assert result == [0, 0, 1, 1, 1] or result == [1, 1, 0, 0, 0]
    assert len(set(result)) == 2


def test_tiny_cluster_that_sounds_like_nobody_keeps_its_own_voice(monkeypatch):
    fake_clusters(monkeypatch, [0, 0, 0, 1])
    man, woman = np.array([1, 0, 0.]), np.array([0, 0, 1.])
    result = group_voices([man, man, man, woman], [60, 60, 60, 13], None, 0.75)
    assert result[3] != result[0]  # a brief second speaker is not merged into the first
    # A single short line is too little to judge a voice by, so it still folds.
    fake_clusters(monkeypatch, [0, 0, 0, 1])
    assert len(set(group_voices([man, man, man, woman], [60, 60, 60, 2], None, 0.75))) == 1


def test_one_person_split_into_two_big_clusters_gets_one_voice(monkeypatch):
    # Short lines can form a cluster of their own; both halves sound alike.
    fake_clusters(monkeypatch, [0, 0, 1, 1, 2, 2])
    woman = np.array([1, 0, 0.])
    waiter_long, waiter_short = np.array([0, 1, 0.]), np.array([0, 0.9, 0.3])
    result = group_voices([woman, woman, waiter_long, waiter_long, waiter_short, waiter_short],
                          [30, 30, 20, 20, 9, 9], None, 0.75)
    assert len(set(result)) == 2
    assert result[2] == result[4]


def test_exact_speaker_count_keeps_small_speakers(monkeypatch):
    fake_clusters(monkeypatch, [0, 0, 1])
    a, b = np.array([1, 0.]), np.array([0, 1.])
    assert len(set(group_voices([a, a, b], [100, 100, 2], 2, 0.75))) == 2


def test_short_lines_use_nearest_voice_and_unmeasurable_lines_carry_on(monkeypatch):
    fake_clusters(monkeypatch, [0, 1])
    a, b = np.array([1, 0.]), np.array([0, 1.])
    result = group_voices([a, b, a * 0.9, None], [50, 50, 0.4, 0.1], None, 0.75)
    assert result[2] == result[0]  # too short to cluster, still matched by voice
    assert result[3] == result[2]  # nothing to measure: same speaker as before


def test_too_little_speech_means_one_speaker():
    assert group_voices([None, np.ones(3)], [0.2, 0.5], None, 0.75) == [0, 0]


def test_second_speaker_of_same_gender_sounds_different():
    segments = [Segment(0, 9, "a", 0), Segment(9, 10, "b", 1), Segment(10, 12, "c", 2)]
    genders = {0: "male", 1: "male", 2: "female"}
    edge = assign_voices(segments, genders, None, "edge")
    assert edge[0] == Voice("ru-RU-DmitryNeural", 0)
    assert edge[1].name == "ru-RU-DmitryNeural" and edge[1].pitch != 0
    assert edge[2] == Voice("ru-RU-SvetlanaNeural", 0)
    piper = assign_voices(segments, genders, None, "piper")
    assert piper[0] == Voice("dmitri", 0) and piper[1] == Voice("denis", 0)


def test_failed_voice_service_raises_instead_of_leaving_silence(tmp_path, monkeypatch):
    import edge_tts

    class Failing:
        def __init__(self, *args, **kwargs):
            pass

        async def save(self, path):
            raise edge_tts.exceptions.NoAudioReceived("no audio")

    async def no_wait(seconds):
        pass

    monkeypatch.setattr(edge_tts, "Communicate", Failing)
    monkeypatch.setattr(dub.asyncio, "sleep", no_wait)
    engine = dub.Edge(tmp_path)
    line = Line("Привет", Voice("x"))

    with pytest.raises(dub.VoiceServiceError):
        asyncio.run(engine._speak(line, asyncio.Semaphore(1)))
    assert not engine._path(line).exists()


def test_punctuation_only_line_is_silent_without_calling_the_service(tmp_path):
    engine = dub.Edge(tmp_path)
    assert len(engine.speak_many([Line("...", Voice("x"))])[0]) == 0


def test_pitch_shift_raises_frequency():
    tone = np.sin(2 * np.pi * 200 * np.arange(22050) / 22050).astype(np.float32)
    shifted = dub.shift_pitch(tone, 1.1)
    assert abs(len(shifted) - 22050 / 1.1) < 2
    assert np.argmax(np.abs(np.fft.rfft(shifted))) * 22050 / len(shifted) == pytest.approx(220, abs=3)


def test_plan_speeds_up_to_fit_and_never_overlaps():
    segments = [Segment(0, 2, "a"), Segment(2, 4, "b"), Segment(10, 12, "c")]
    placed = plan(segments, [3.0, 4.0, 1.0], 20, max_speed=1.6)
    for one, two in zip(placed, placed[1:]):
        assert one.start + one.length <= two.start
    assert placed[1].start <= 2 + dub.SYNC_LAG  # still in sync
    assert placed[2].start == 10  # the pause brings the dub back to the original timing


def test_a_phrase_is_spoken_at_one_steady_speed():
    # Old timing rushed the first line to 1.54x and left the second at 1.0x.
    segments = [Segment(0, 2, "a"), Segment(2, 4, "b"), Segment(10, 12, "c")]
    placed = plan(segments, [3.0, 4.0, 1.0], 20, max_speed=1.6)
    assert placed[0].speed == placed[1].speed
    assert 1.2 < placed[0].speed < 1.25  # just enough to keep line b within the lag
    assert placed[2].speed == 1.0


def test_lines_start_early_only_when_that_helps():
    calm = plan([Segment(5, 7, "a")], [1.0], 20, max_speed=1.6)
    assert calm[0].start == 5  # no reason to move it
    tight = plan([Segment(5, 7, "a"), Segment(7, 9, "b")], [2.5, 2.0], 20, max_speed=1.6)
    assert tight[0].start == pytest.approx(5 - dub.SYNC_LEAD)


class FakeEngine:
    """Speaks at 10 chars/s, while the guess assumes a different rate."""

    rate = 1000

    def __init__(self, guess):
        self.guess = guess
        self.calls = []

    def chars_per_second(self, voice):
        return self.guess

    def speak_many(self, lines):
        self.calls.append(lines)
        return [np.ones(int(len(l.text) / 10 / l.speed * self.rate), dtype=np.float32) for l in lines]


def test_good_speed_guess_needs_only_one_pass(monkeypatch):
    monkeypatch.setattr(dub, "stretch", lambda a, t, r: a[:int(len(a) / t)])
    segments = [Segment(0, 2, "x" * 30), Segment(2, 4, "y" * 15)]
    engine = FakeEngine(guess=10)
    track, placed = dub.synthesize(segments, {0: Voice("v")}, 10, 1.6, engine)
    assert len(engine.calls) == 1
    assert placed[0].speed == pytest.approx(1.22, abs=0.02)


def test_bad_speed_guess_speaks_the_line_again(monkeypatch):
    monkeypatch.setattr(dub, "stretch", lambda a, t, r: a[:int(len(a) / t)])
    segments = [Segment(0, 2, "x" * 30), Segment(2, 4, "y" * 15)]
    engine = FakeEngine(guess=30)  # thinks the voice is 3x faster than it is
    dub.synthesize(segments, {0: Voice("v")}, 10, 1.6, engine)
    assert len(engine.calls) == 2
    assert engine.calls[1][0].speed == pytest.approx(1.22, abs=0.02)


def test_lines_running_past_the_end_are_kept_not_crashing(monkeypatch):
    monkeypatch.setattr(dub, "stretch", lambda a, t, r: a[:int(len(a) / t)])
    # Dense speech up to the last moment: the dub falls further and further behind.
    segments = [Segment(i * 2, i * 2 + 2, "x" * 40) for i in range(8)]
    track, placed = dub.synthesize(segments, {0: Voice("v")}, 16.2, 1.6, FakeEngine(guess=10))
    last_end = placed[-1].start + placed[-1].length
    assert last_end > 17.2  # starts after the old track would have ended
    assert len(track) >= last_end * FakeEngine.rate
    assert track[int(placed[-1].start * FakeEngine.rate) + 1] == 1  # and it was not dropped


class SilentEngine(FakeEngine):
    def __init__(self, silent_texts):
        super().__init__(guess=10)
        self.silent = silent_texts

    def speak_many(self, lines):
        clips = super().speak_many(lines)
        return [clip[:0] if line.text in self.silent else clip for line, clip in zip(lines, clips)]


def test_voice_that_returns_no_audio_fails_instead_of_a_silent_dub():
    segments = [Segment(i * 3, i * 3 + 2, f"line {i}") for i in range(10)]
    everything = {s.text for s in segments}
    with pytest.raises(dub.VoiceServiceError):
        dub.synthesize(segments, {0: Voice("v")}, 40, 1.6, SilentEngine(everything))
    # One unspeakable line out of ten is reported but does not stop the video.
    dub.synthesize(segments, {0: Voice("v")}, 40, 1.6, SilentEngine({"line 3"}))


def test_short_pauses_stay_ducked_even_when_speech_starts_at_zero():
    segments = [Segment(0.0, 2, "a"), Segment(2.8, 5, "b"), Segment(5.8, 8, "c")]
    placed = [dub.Placement(s.start, 1.0, s.end - s.start) for s in segments]
    gain = dub.duck_envelope(segments, placed, 10, 0.15)
    pause = int(2.4 * dub.ENVELOPE_RATE)
    assert gain[pause] == pytest.approx(0.15)
    assert gain[-1] == pytest.approx(1.0)  # back to full volume after the speech
    quiet = dub.duck_envelope([Segment(5, 6, "a")], [dub.Placement(5, 1, 1)], 10, 0.15)
    assert quiet[0] == pytest.approx(1.0)  # the start of the video is not faded in


def make_media(path, seconds=1.0, channels=2, video=True, sound="aac", amplitude=0.5,
               cover=False):
    """A small test file, written with PyAV: a 440 Hz tone, with a black picture if asked."""
    import av

    rate, layout = 48000, "mono" if channels == 1 else "stereo"
    with av.open(str(path), "w") as out:
        audio = out.add_stream(sound, rate=rate, layout=layout)
        picture = None
        if video or cover:
            picture = out.add_stream("mjpeg" if cover else "libx264", rate=25)
            picture.width = picture.height = 64
            picture.pix_fmt = "yuvj420p" if cover else "yuv420p"
            if cover:
                picture.disposition = av.stream.Disposition.attached_pic
        if picture:
            black = av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), np.uint8), format="rgb24")
            for i in range(1 if cover else int(seconds * 25)):
                frame = black.reformat(format=picture.pix_fmt)
                frame.pts = i
                out.mux(picture.encode(frame))
            out.mux(picture.encode(None))
        t = np.arange(int(seconds * rate)) / rate
        wave = (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        frame = av.AudioFrame.from_ndarray(np.repeat(wave[None], channels, axis=0),
                                           format="fltp", layout=layout)
        frame.sample_rate, frame.pts = rate, 0
        out.mux(audio.encode(frame))
        out.mux(audio.encode(None))


@pytest.mark.parametrize("channels", [1, 2])
def test_mix_keeps_the_original_level_where_nobody_speaks(tmp_path, channels):
    from vtranslate import media

    source = tmp_path / "source.wav"
    make_media(source, 2.0, channels, video=False, sound="pcm_f32le")
    video = tmp_path / "video.mp4"
    make_media(video, 2.0)
    out = tmp_path / "out.mp4"
    envelope = np.ones(300, dtype=np.float32)  # no ducking at all
    dub.render(video, source, np.zeros(44100, dtype=np.float32), 22050, envelope, out)
    mixed = np.concatenate(list(media.decode(out, 48000, "stereo")), axis=1)[:, 4800:-4800]
    rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
    assert rms(mixed) == pytest.approx(0.5 / np.sqrt(2), rel=0.05)  # same as the source


def test_mix_keeps_speech_running_past_the_end_of_the_video(tmp_path):
    import av

    video = tmp_path / "video.mp4"
    make_media(video, 1.0)
    out = tmp_path / "out.mkv"
    voice = np.full(3 * 22050, 0.1, dtype=np.float32)
    dub.render(video, video, voice, 22050, np.ones(200, dtype=np.float32), out)
    with av.open(str(out)) as container:
        streams = {s.type: s for s in container.streams}
        assert streams["video"].codec_context.name == "h264"  # copied, not re-encoded
        assert streams["audio"].metadata.get("language") == "rus"
        assert container.duration / av.time_base == pytest.approx(3.0, abs=0.1)


def test_speech_audio_is_mono_at_16khz(tmp_path):
    from vtranslate import fetch, media

    video = tmp_path / "video.mp4"
    make_media(video, 1.0, channels=2, amplitude=0.25)
    fetch.extract_speech_audio(video, tmp_path / "speech.wav")
    samples = media.read_wav(tmp_path / "speech.wav")
    assert len(samples) == pytest.approx(16000, abs=1100)  # the encoder pads a little
    # FFmpeg's downmix: (left + right) / sqrt(2), so two 0.25 sides give 0.25 * sqrt(2)
    level = float(np.sqrt(np.mean(samples[1600:-1600] ** 2)))
    assert level == pytest.approx(0.25, rel=0.1)


def test_stretch_changes_length_but_not_pitch():
    rate = 22050
    t = np.arange(rate) / rate
    sound = np.sin(2 * np.pi * 300 * t).astype(np.float32)
    faster = dub.stretch(sound, 1.25, rate)
    assert len(faster) == pytest.approx(rate / 1.25, rel=0.03)
    pitch = np.argmax(np.abs(np.fft.rfft(faster))) * rate / len(faster)
    assert pitch == pytest.approx(300, abs=5)


def test_identical_lines_are_requested_once(tmp_path, monkeypatch):
    import edge_tts

    calls = []

    class Counting:
        def __init__(self, text, *args, **kwargs):
            calls.append(text)

        async def save(self, path):
            Path(path).write_bytes(b"x")

    monkeypatch.setattr(edge_tts, "Communicate", Counting)
    engine = dub.Edge(tmp_path)
    monkeypatch.setattr(engine, "_decode", lambda line: np.zeros(0, dtype=np.float32))
    engine.speak_many([Line("Спасибо.", Voice("v"))] * 5)
    assert calls == ["Спасибо."]


def test_long_gaps_never_exceed_whispers_token_limit(monkeypatch):
    import faster_whisper.vad

    budgets = []

    class Model:
        def transcribe(self, clip, **options):
            budgets.append(options["max_new_tokens"])
            return [], None

    hour = 3600
    monkeypatch.setattr(faster_whisper.vad, "get_speech_timestamps",
                        lambda samples, options: [{"start": 0, "end": hour * speech.RATE}])
    speech._recover_skipped(Model(), np.zeros(10, dtype=np.float32), [], "en")
    assert budgets and max(budgets) + 3 <= 448  # 3 prompt tokens + budget within Whisper's limit


def test_incomplete_model_folder_is_downloaded_again():
    attempts = []

    def load(**options):
        attempts.append(options)
        if options.get("local_files_only"):
            raise RuntimeError("Unable to open file 'model.bin'")
        return "model"

    assert speech.cached_first(load) == "model"
    assert attempts == [{"local_files_only": True}, {}]


def test_file_names_fit_in_255_bytes():
    from vtranslate.cli import NAME_BYTES, safe_name

    for title in ["語" * 150, "Я" * 150, "😀" * 100, "a" * 400]:
        name = safe_name(title)
        assert name and len(name.encode()) <= NAME_BYTES
        assert len(f".{name} [ru].tmp.mp4".encode()) <= 255
    assert safe_name("...") == "video"


def test_resuming_at_another_quality_uses_a_different_job_folder():
    from vtranslate.fetch import job_id

    url = "https://example.com/watch?v=1"
    assert job_id(url, 720) != job_id(url, 1080)
    assert job_id(url, 720) == job_id(url, 720)


def test_files_are_inspected_without_ffprobe(tmp_path):
    from vtranslate import fetch

    video = tmp_path / "video.mp4"
    make_media(video)
    assert fetch.video_codec(video) == "h264" and fetch.has_audio(video)
    song = tmp_path / "song.mp3"
    make_media(song, channels=1, video=False, sound="libmp3lame", cover=True)
    assert fetch.has_audio(song) and not fetch.has_video(song)  # cover art is not a video
    with pytest.raises(fetch.SourceError):
        fetch.has_audio(tmp_path / "missing.mp4")


def test_one_bad_source_does_not_stop_the_batch(monkeypatch, capsys):
    from vtranslate import cli
    from vtranslate.fetch import SourceError

    done = []

    def fake_run(source, args):
        if source == "bad":
            raise SourceError("the speech is already in Russian")
        done.append(source)
        return Path(source)

    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr(cli, "prepare_translation", lambda args: None)
    monkeypatch.setattr(sys, "argv", ["vtranslate", "bad", "good"])
    with pytest.raises(SystemExit) as exit_:
        cli.main()
    assert exit_.value.code == 1
    assert done == ["good"]
    assert "already in Russian" in capsys.readouterr().err


def test_bad_settings_are_refused_before_any_work():
    from vtranslate.cli import parse_args

    for bad in (["--from", "english"], ["--from", "ru"], ["--max-speed", "0"],
                ["--original-volume", "2"], ["--speakers", "0"], ["-q", "0"]):
        with pytest.raises(SystemExit):
            parse_args(["x", *bad])
    assert parse_args(["x", "--from", "my"]).source_language == "my"


class FakeDeepL:
    """Stands in for deepl.DeepLClient; records what would have been sent."""

    requests = []
    used, limit = 0, 500_000
    sources = ["EN", "DE", "ES"]
    error = None

    def __init__(self, key, **options):
        self.key = key

    def get_source_languages(self):
        from types import SimpleNamespace

        return [SimpleNamespace(code=code) for code in self.sources]

    def get_usage(self):
        from types import SimpleNamespace

        if self.error:
            raise self.error
        return SimpleNamespace(character=SimpleNamespace(valid=True, count=self.used, limit=self.limit))

    def translate_text(self, texts, **options):
        from types import SimpleNamespace

        if self.error:
            raise self.error
        FakeDeepL.requests.append((list(texts), options))
        return [SimpleNamespace(text=f"RU<{t}>") for t in texts]


@pytest.fixture
def fake_deepl(monkeypatch, tmp_path):
    import deepl

    from vtranslate import translate as module

    FakeDeepL.requests, FakeDeepL.used, FakeDeepL.error = [], 0, None
    FakeDeepL.sources = ["EN", "DE", "ES"]
    monkeypatch.setattr(deepl, "DeepLClient", FakeDeepL)
    monkeypatch.setenv("DEEPL_AUTH_KEY", "test-key:fx")
    monkeypatch.setattr(module, "KEY_FILE", tmp_path / "deepl-key")
    return module


def texts_of(count):
    return [f"line {i}." for i in range(count)]


def test_deepl_keeps_order_and_sends_the_conversation_as_context(fake_deepl):
    russian = fake_deepl.translate(texts_of(60), "en", "My video")
    assert russian == [f"RU<line {i}.>" for i in range(60)]
    assert len(FakeDeepL.requests) == -(-60 // fake_deepl.CHUNK)  # in chunks, not line by line
    texts, options = FakeDeepL.requests[1]
    assert options["target_lang"] == "RU" and options["source_lang"] == "EN"
    assert "My video" in options["context"]
    assert f"line {fake_deepl.CHUNK - 1}." in options["context"]  # what came just before
    assert f"line {2 * fake_deepl.CHUNK}." in options["context"]  # and what comes next


@pytest.mark.parametrize("problem, message", [
    ("no key", "--set-deepl-key"),
    ("rejected", "rejected the key"),
    ("used up", "allowance is used up"),
    ("too long", "only 100 of"),
    ("offline", "could not reach DeepL"),
    ("language", "doesn't translate"),
])
def test_each_deepl_problem_gives_a_clear_reason(fake_deepl, monkeypatch, problem, message):
    import deepl

    if problem == "no key":
        monkeypatch.delenv("DEEPL_AUTH_KEY")
    elif problem == "rejected":
        FakeDeepL.error = deepl.AuthorizationException("bad key")
    elif problem == "used up":
        FakeDeepL.used = FakeDeepL.limit
    elif problem == "too long":
        FakeDeepL.used = FakeDeepL.limit - 100
    elif problem == "offline":
        FakeDeepL.error = deepl.ConnectionException("no route")
    else:
        FakeDeepL.sources = ["DE"]
    with pytest.raises(fake_deepl.TranslationError, match=message):
        fake_deepl.translate(texts_of(30), "en")
    assert not FakeDeepL.requests  # nothing is half-translated


def test_saved_key_is_private(tmp_path, monkeypatch):
    import os
    import stat

    from vtranslate import translate as module

    monkeypatch.setattr(module, "KEY_FILE", tmp_path / "vt" / "deepl-key")
    monkeypatch.delenv("DEEPL_AUTH_KEY", raising=False)
    path = module.save_key("  abc:fx \n")
    assert module.key() == "abc:fx"
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_first_run_asks_for_a_key_once_and_saves_it(fake_deepl, monkeypatch):
    import getpass

    from vtranslate import cli

    monkeypatch.delenv("DEEPL_AUTH_KEY")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    asked = []
    monkeypatch.setattr(getpass, "getpass", lambda prompt: asked.append(prompt) or "new-key:fx")
    cli.prepare_translation(cli.parse_args(["video.mp4"]))
    assert fake_deepl.key() == "new-key:fx" and len(asked) == 1
    cli.prepare_translation(cli.parse_args(["video.mp4"]))  # second run: no question
    assert len(asked) == 1


def test_without_a_terminal_it_explains_instead_of_asking(fake_deepl, monkeypatch):
    from vtranslate import cli

    monkeypatch.delenv("DEEPL_AUTH_KEY")
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(SystemExit, match="--set-deepl-key"):
        cli.prepare_translation(cli.parse_args(["video.mp4"]))


def test_a_used_up_allowance_stops_before_any_slow_work(fake_deepl):
    from vtranslate import cli

    FakeDeepL.used = FakeDeepL.limit
    with pytest.raises(SystemExit, match="used up"):
        cli.prepare_translation(cli.parse_args(["video.mp4"]))


def test_defaults_are_fast_and_best_switches_everything():
    from vtranslate.cli import parse_args

    fast = parse_args(["x"])
    assert (fast.model, fast.tts, fast.best) == ("base", "piper", False)
    best = parse_args(["x", "--best"])
    assert (best.model, best.tts, best.best) == ("base", "auto", True)
    chosen = parse_args(["x", "--best", "-m", "tiny", "--tts", "piper"])
    assert (chosen.model, chosen.tts) == ("tiny", "piper")
