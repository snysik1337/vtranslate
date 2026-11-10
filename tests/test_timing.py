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


def test_speed_changes_gently_inside_a_phrase():
    # Old timing rushed the first line to 1.54x and left the second at 1.0x.
    segments = [Segment(0, 2, "a"), Segment(2, 4, "b"), Segment(10, 12, "c")]
    placed = plan(segments, [2.9, 2.0, 1.0], 20, max_speed=1.6)
    # Just enough to keep line b within the lag.
    assert placed[0].speed == pytest.approx(2.9 / (2 + dub.SYNC_LAG - dub.GAP), abs=0.01)
    # Line b has room of its own, so it eases off, but only by one step.
    assert placed[1].speed == pytest.approx(placed[0].speed - dub.MAX_STEP)
    assert placed[2].speed == dub.CALM_SPEED  # room to spare: a calmer pace


def test_lines_never_start_before_the_speaker():
    calm = plan([Segment(5, 7, "a")], [1.0], 20, max_speed=1.6)
    assert calm[0].start == 5
    # Even when an early start would save speed.
    tight = plan([Segment(5, 7, "a"), Segment(7, 9, "b")], [2.5, 2.0], 20, max_speed=1.6)
    assert tight[0].start == 5 and tight[0].speed > 1.0


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
    assert placed[0].speed == pytest.approx(3.0 / (2 + dub.SYNC_LAG - dub.GAP), abs=0.02)


def test_bad_speed_guess_speaks_the_line_again(monkeypatch):
    monkeypatch.setattr(dub, "stretch", lambda a, t, r: a[:int(len(a) / t)])
    segments = [Segment(0, 2, "x" * 30), Segment(2, 4, "y" * 15)]
    engine = FakeEngine(guess=30)  # thinks the voice is 3x faster than it is
    dub.synthesize(segments, {0: Voice("v")}, 10, 1.6, engine)
    assert len(engine.calls) == 2
    assert engine.calls[1][0].speed == pytest.approx(3.0 / (2 + dub.SYNC_LAG - dub.GAP), abs=0.02)


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
    media.render(video, source, np.zeros(44100, dtype=np.float32), 22050, envelope, 100, out)
    mixed = np.concatenate(list(media.decode(out, 48000, "stereo")), axis=1)[:, 4800:-4800]
    rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
    assert rms(mixed) == pytest.approx(0.5 / np.sqrt(2), rel=0.05)  # same as the source


def test_mix_keeps_speech_running_past_the_end_of_the_video(tmp_path):
    import av

    from vtranslate import media

    video = tmp_path / "video.mp4"
    make_media(video, 1.0)
    out = tmp_path / "out.mkv"
    voice = np.full(3 * 22050, 0.1, dtype=np.float32)
    media.render(video, video, voice, 22050, np.ones(200, dtype=np.float32), 100, out)
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


def test_usage_note_after_translation_never_fails_the_dub(monkeypatch, capsys):
    from vtranslate import cli
    from vtranslate import translate as deepl

    def used_up(value=None):
        raise deepl.TranslationError("this month's DeepL allowance is used up")

    monkeypatch.setattr(deepl, "check_ready", used_up)
    cli.print_usage("  ", required=False)  # the translation that used it up is saved
    with pytest.raises(deepl.TranslationError):
        cli.print_usage()  # before starting, though, it must stop the run


def test_matroska_packets_without_decode_times_are_kept(tmp_path):
    import av

    from vtranslate import media

    video = tmp_path / "video.mkv"  # libx264 uses B-frames, which leave some dts unset
    make_media(video, 2.0)
    sound = tmp_path / "sound.wav"
    make_media(sound, 2.0, video=False, sound="pcm_f32le")
    out = tmp_path / "out.mp4"
    media.render(video, sound, np.zeros(100, dtype=np.float32), 22050,
                 np.ones(300, dtype=np.float32), 100, out)
    count = lambda path: sum(1 for p in av.open(str(path)).demux(video=0) if p.size)
    assert count(out) == count(video) == 50


def test_media_errors_stay_with_one_source():
    from vtranslate import media

    with pytest.raises(media.MediaError):  # a RuntimeError, so only this source fails
        media.stretch(np.ones(22050, dtype=np.float32), 0.0, 22050)


def test_each_file_is_opened_once_for_inspection(tmp_path, monkeypatch):
    import av

    from vtranslate import fetch

    video = tmp_path / "video.mp4"
    make_media(video)
    opened = []
    real = av.open
    monkeypatch.setattr(av, "open", lambda *a, **k: opened.append(a) or real(*a, **k))
    assert fetch.has_audio(video) and fetch.has_video(video) and fetch.video_codec(video) == "h264"
    assert len(opened) == 1


@pytest.mark.parametrize("problem, expected", [
    (EOFError, ""),  # Ctrl+D: cancel the prompt
    (UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"), "text editor"),
    (OSError("no terminal"), "text editor"),
])
def test_a_failed_key_prompt_is_explained_not_a_crash(monkeypatch, capsys, problem, expected):
    import getpass

    from vtranslate import cli

    def failing(prompt=""):
        raise problem

    monkeypatch.setattr(getpass, "getpass", failing)
    assert cli.ask_for_key() is False
    out = capsys.readouterr().out
    assert expected in out and "DEEPL_AUTH_KEY=" not in out  # never on the command line


def test_ctrl_c_at_the_key_prompt_stops_vtranslate(monkeypatch, fake_deepl):
    import getpass

    from vtranslate import cli

    def interrupted(prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr(getpass, "getpass", interrupted)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    started = []
    monkeypatch.setattr(cli, "run", lambda source, args: started.append(source))
    monkeypatch.setattr(sys, "argv", ["vtranslate", "--set-deepl-key", "https://example.com/v"])
    with pytest.raises(SystemExit) as exit_:
        cli.main()
    assert exit_.value.code == 130 and not started  # no dub after Ctrl+C


def test_ctrl_c_during_a_batch_exits_with_130(monkeypatch, capsys):
    from vtranslate import cli

    done = []

    def fake_run(source, args):
        if source == "b":
            raise KeyboardInterrupt
        done.append(source)
        return Path(source)

    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr(cli, "prepare_translation", lambda args: None)
    monkeypatch.setattr(sys, "argv", ["vtranslate", "a", "b", "c"])
    with pytest.raises(SystemExit) as exit_:
        cli.main()
    assert exit_.value.code == 130 and done == ["a"]
    assert "run the same command again" in capsys.readouterr().err


def test_rerunning_skips_finished_videos_unless_settings_change(tmp_path, monkeypatch, capsys):
    from vtranslate import cli

    monkeypatch.setattr(cli, "CACHE", tmp_path / "jobs")
    source = tmp_path / "talk.mp4"
    source.write_bytes(b"video")
    dubbed = []

    def fake_dub(source, args, work):
        dubbed.append(args.speakers)
        output = args.output / "talk [ru].mp4"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"dub")
        return output

    monkeypatch.setattr(cli, "_dub", fake_dub)
    args = lambda *extra: cli.parse_args(["-o", str(tmp_path / "out"), *extra, str(source)])
    first = cli.run(str(source), args())
    with pytest.raises(cli.AlreadyDubbed):
        cli.run(str(source), args())  # same settings: skipped, not paid for again
    cli.run(str(source), args("--speakers", "3"))  # new settings: dubbed again
    first.unlink()
    cli.run(str(source), args("--speakers", "3"))  # deleted: dubbed again
    assert dubbed == [None, 3, 3]
    monkeypatch.setattr(sys, "argv", ["vtranslate", "-o", str(tmp_path / "out"),
                                      "--speakers", "3", str(source)])
    monkeypatch.setattr(cli, "prepare_translation", lambda args: None)
    with pytest.raises(SystemExit):
        cli.main()
    out = capsys.readouterr().out
    assert "Already dubbed" in out and "Done in" not in out


def test_thread_pool_drops_queued_work_on_ctrl_c():
    import time

    started = []

    def slow(item):
        started.append(item)
        if item == 0:
            raise KeyboardInterrupt
        time.sleep(0.05)

    with pytest.raises(KeyboardInterrupt):
        dub.parallel(slow, range(400), threads=2)
    assert len(started) < 20


def test_progress_counter_counts_every_line_across_threads(capsys):
    count = dub.Counter(2000)
    dub.parallel(count, range(2000), threads=8)
    assert count.done == 2000


def test_cancelling_a_new_key_keeps_the_existing_one(monkeypatch, capsys, fake_deepl):
    import getpass

    from vtranslate import cli

    monkeypatch.setattr(getpass, "getpass", lambda prompt="": "")  # Enter: cancel
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    cli.prepare_translation(cli.parse_args(["--set-deepl-key", "https://example.com/v"]))
    assert "Keeping the key you already have" in capsys.readouterr().out


def test_find_gaps_is_fast_and_ignores_words_before_a_span():
    import time

    words = [Word(i * 0.4, i * 0.4 + 0.3, " w") for i in range(30000)]
    spans = [(i * 4.0, i * 4.0 + 3.5) for i in range(3000)]
    began = time.time()
    assert speech.find_gaps(spans, words) == []
    assert time.time() - began < 0.5
    # A word that ends before a span starts must not hide a gap inside it.
    assert speech.find_gaps([(10.0, 14.0)], [Word(1.0, 2.0, " a"), Word(13.5, 13.9, " b")]) \
        == [(10.0, 13.5)]


def test_cut_off_downloads_are_never_kept(tmp_path, monkeypatch):
    import io
    import urllib.request

    class Short(io.BytesIO):
        headers = {"Content-Length": "100"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: Short(b"x" * 40))
    dest = tmp_path / "voice.onnx"
    with pytest.raises(OSError, match="cut short"):
        speech.download_file("https://example.com/voice.onnx", dest)
    assert not dest.exists() and not list(tmp_path.iterdir())


def test_voices_talk_calmer_when_there_is_room_and_speed_up_only_when_needed():
    roomy = plan([Segment(0, 3, "a")], [2.0], 20, max_speed=1.6)
    assert roomy[0].speed == dub.CALM_SPEED  # 8% slower than natural
    # Line a has 2.9 s to say in 2 s, and may run on only SYNC_OVERRUN after its speaker.
    snug = plan([Segment(0, 2, "a"), Segment(2.5, 4, "b")], [2.9, 1.0], 20, max_speed=1.6)
    assert snug[0].speed == pytest.approx(2.9 / (2 + dub.SYNC_OVERRUN), abs=0.01)


def test_long_pauses_inside_a_line_are_shortened():
    rate = 1000
    word = np.ones(300, dtype=np.float32) * 0.5
    line = np.concatenate([np.zeros(50), word, np.zeros(700), word, np.zeros(100), word, np.zeros(50)])
    out = dub.shorten_pauses(line.astype(np.float32), rate)
    assert len(out) == len(line) - (700 - 250)  # only the long inner pause is cut
    assert out[:50].sum() == 0 and out[-50:].sum() == 0  # edges are left to trim()


def test_pronunciation_fixes_add_yo_and_stress_but_keep_ambiguous_words():
    from vtranslate.stress import ACCENT, fix

    assert fix("Еще раз: учетной записи нет, он пойдет.") == "Ещё раз: учётной записи нет, он пойдёт."
    assert fix("Перед этим что-то сломалось.") == f"Пе{ACCENT}ред этим что{ACCENT}то сломалось."
    assert fix("ПЕРЕД") == f"ПЕ{ACCENT}РЕД"
    unchanged = "Все воды и сети, передать, перешел, Linux 2.0"  # meaning-dependent or unlisted
    assert fix(unchanged) == unchanged


def test_every_pronunciation_fix_is_a_real_change():
    from vtranslate.stress import FIXES

    for plain, fixed in FIXES.items():
        assert plain != fixed and "+" not in fixed and "ё" not in plain


def test_calm_pace_never_starts_lines_before_the_speaker():
    # Dense talk where 0.98x fits but 0.92x does not: no early start just to be calmer.
    segments = [Segment(i * 2.0, i * 2.0 + 1.9, "a") for i in range(10)]
    placed = plan(segments, [1.9] * 10, 30, max_speed=1.6)
    assert all(p.start >= s.start - 1e-9 for s, p in zip(segments, placed))


def test_a_voice_that_fails_for_other_reasons_is_not_deleted(tmp_path, monkeypatch):
    import onnxruntime

    monkeypatch.setattr(dub, "MODELS", tmp_path)
    model = tmp_path / "piper" / "ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"model")
    model.with_name(model.name + ".json").write_text("{}")

    def out_of_memory(*args, **kwargs):
        raise RuntimeError("Failed to allocate memory")

    monkeypatch.setattr(onnxruntime, "InferenceSession", out_of_memory)
    with pytest.raises(RuntimeError, match="could not be loaded"):
        dub.Piper()._load("dmitri")
    assert model.exists()  # kept: downloading it again would not help

    def damaged(*args, **kwargs):
        raise RuntimeError("[ONNXRuntimeError] : 7 : INVALID_PROTOBUF : Load model failed")

    monkeypatch.setattr(onnxruntime, "InferenceSession", damaged)
    with pytest.raises(RuntimeError, match="damaged"):
        dub.Piper()._load("dmitri")
    assert not model.exists()  # fetched afresh next time


def test_beret_and_other_meanings_are_left_alone_and_edge_gets_only_yo():
    from vtranslate.stress import ACCENT, fix

    assert fix("красный берет") == "красный берет"
    assert fix("Перед этим он пойдет", marks=False) == "Перед этим он пойдёт"
    assert ACCENT not in fix("Перед что-то", marks=False)


def test_a_finished_batch_reruns_without_deepl(tmp_path, monkeypatch, capsys):
    from vtranslate import cli

    monkeypatch.setattr(cli, "CACHE", tmp_path / "jobs")
    source = tmp_path / "talk.mp4"
    source.write_bytes(b"video")
    out = tmp_path / "out"

    def fake_dub(source, args, work):
        path = args.output / "talk [ru].mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"dub")
        return path

    monkeypatch.setattr(cli, "_dub", fake_dub)
    cli.run(str(source), cli.parse_args(["-o", str(out), str(source)]))

    def offline(args):
        raise SystemExit("error: could not reach DeepL")

    monkeypatch.setattr(cli, "prepare_translation", offline)
    monkeypatch.setattr(sys, "argv", ["vtranslate", "-o", str(out), str(source)])
    with pytest.raises(SystemExit) as exit_:
        cli.main()
    assert exit_.value.code == 0 and "Already dubbed" in capsys.readouterr().out


def test_two_downloads_of_one_file_never_share_a_temp_file(tmp_path, monkeypatch):
    import io
    import threading
    import urllib.request

    seen = []

    class Slow(io.BytesIO):
        headers = {"Content-Length": "4"}

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, *args):
            seen.extend(p.name for p in tmp_path.glob("*.part"))
            return super().read(*args)

    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: Slow(b"data"))
    dest = tmp_path / "voice.onnx"
    threads = [threading.Thread(target=speech.download_file, args=("u", dest)) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert dest.read_bytes() == b"data" and not list(tmp_path.glob("*.part"))
    assert all(name.startswith("voice.onnx.") for name in seen)


def test_soft_speech_is_not_mistaken_for_a_pause():
    rate = 1000
    soft = np.full(600, 0.004, dtype=np.float32)  # a quiet line: below 0.005 throughout
    line = np.concatenate([np.full(100, 0.3), soft, np.full(100, 0.3)]).astype(np.float32) * 0.01
    assert len(dub.shorten_pauses(line, rate)) == len(line)


def _fake_youtube(monkeypatch, tmp_path, picture_waits=False):
    """A stand-in for yt-dlp: audio and picture as separate files, counting lookups."""
    import time

    import yt_dlp

    audio_file, video_file = tmp_path / "a.m4a", tmp_path / "v.mp4"
    make_media(audio_file, video=False)
    make_media(video_file)
    lookups = []

    class FakeYDL:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def extract_info(self, url, download=False, process=True):
            lookups.append(url)
            return {"id": "x", "title": "Talk", "formats": [{"vcodec": "h264"}, {"vcodec": "none"}]}

        def process_ie_result(self, info, download=True):
            video = self.options["format"] == "bv*"
            while video and picture_waits:
                for hook in self.options["progress_hooks"]:
                    hook({"status": "downloading"})
                time.sleep(0.01)
            target = Path(self.options["outtmpl"].replace("%(ext)s", "mp4" if video else "m4a"))
            target.write_bytes((video_file if video else audio_file).read_bytes())
            return {**info, "requested_downloads": [{"filepath": str(target)}]}

    monkeypatch.setattr(yt_dlp, "YoutubeDL", FakeYDL)
    return lookups


def test_the_picture_download_reuses_the_first_lookup(tmp_path, monkeypatch):
    from vtranslate import fetch

    lookups = _fake_youtube(monkeypatch, tmp_path)
    work = tmp_path / "job"
    work.mkdir()
    got = fetch.download("https://www.youtube.com/watch?v=x", work, 720, None)
    assert fetch.has_video(got.video_path())
    assert len(lookups) == 1  # YouTube looked up once, not once per download


def test_a_failed_dub_stops_the_picture_download(tmp_path, monkeypatch):
    from vtranslate import fetch

    _fake_youtube(monkeypatch, tmp_path, picture_waits=True)
    work = tmp_path / "job"
    work.mkdir()
    got = fetch.download("https://www.youtube.com/watch?v=x", work, 720, None)
    got.cancel()
    got.pending.thread.join(timeout=2)
    assert got.pending.done and "the dub stopped" in str(got.pending.error)  # stopped, not finished


def test_one_overlong_line_does_not_rush_the_whole_phrase():
    # Twenty back-to-back lines; only the first needs far more time than it has.
    segs = [Segment(i * 3.1, i * 3.1 + 3, "x") for i in range(20)]
    placed = dub.plan(segs, [8.0] + [2.5] * 19, 70, 1.6)
    speeds = [round(p.speed, 2) for p in placed]
    assert speeds[0] == 1.6
    assert speeds.count(1.6) <= 4  # the long line and the few catching up after it
    assert speeds[-1] == dub.CALM_SPEED
    assert all(abs(a - b) <= dub.MAX_STEP + 0.011 for a, b in zip(speeds, speeds[1:]))
    for before, after in zip(placed, placed[1:]):
        assert after.start >= before.start + before.length  # never two lines at once
    assert all(p.start <= s.start + dub.SYNC_LAG + 1e-9 for p, s in zip(placed[4:], segs[4:]))


def test_a_late_phrase_only_rushes_the_lines_that_catch_up():
    segs = [Segment(0, 3, "x")] + [Segment(4.2 + i * 3.1, 7.2 + i * 3.1, "x") for i in range(10)]
    placed = dub.plan(segs, [9.0] + [2.5] * 10, 40, 1.6)
    speeds = [round(p.speed, 2) for p in placed]
    assert speeds[:2] == [1.6, 1.6]
    assert speeds[-1] == dub.CALM_SPEED and speeds.count(1.6) <= 4
    assert speeds == sorted(speeds, reverse=True)  # it eases off step by step


def test_split_phrases_are_never_later_than_rushing_everything():
    import random

    rng = random.Random(7)
    for _ in range(200):
        at, segs, natural = 0.0, [], []
        for _ in range(rng.randint(1, 40)):
            at += rng.choice([0.1, 0.3, 0.8, 1.5, 4.0])
            length = rng.uniform(0.8, 6)
            segs.append(Segment(at, at + length, "x"))
            natural.append(length * rng.choice([0.6, 0.9, 1.1, 1.5, 2.5]))
            at += length
        placed = dub.plan(segs, natural, at, 1.6)
        cursor, worst = 0.0, 0.0
        for seg, length in zip(segs, natural):  # everything at top speed
            start = max(cursor, seg.start)
            worst = max(worst, start - seg.start)
            cursor = start + length / 1.6 + dub.GAP
        assert max(p.start - s.start for p, s in zip(placed, segs)) <= max(worst, dub.SYNC_LAG) + 1e-6
        assert all(dub.CALM_SPEED - 1e-9 <= p.speed <= 1.6 + 1e-9 for p in placed)
        for before, after in zip(placed, placed[1:]):
            assert after.start >= before.start + before.length - 1e-9


def test_a_line_spoken_too_fast_is_slowed_to_the_plan(monkeypatch):
    class Quick:
        rate = 16000

        def chars_per_second(self, voice):
            return 20 / 2.6  # the guess: 20 characters take 2.6 s

        def speak_many(self, lines):
            # The voice turns out quicker than guessed: 2.3 s for 20 characters.
            return [np.full(int(2.3 * len(line.text) / 20 / line.speed * self.rate), 0.1,
                            dtype=np.float32) for line in lines]

    stretched = []
    monkeypatch.setattr(dub, "stretch", lambda audio, tempo, rate: stretched.append(tempo) or audio)
    # The next line follows at once, so the first was guessed to need speeding up.
    segs = [Segment(0.0, 1.9, "х" * 20), Segment(2.0, 3.0, "х" * 4)]
    dub.synthesize(segs, {0: dub.Voice("dmitri")}, 10, 1.6, Quick())
    assert stretched and 1 / dub.MAX_STRETCH <= stretched[0] < 0.98  # slowed, not left fast


def test_a_tight_spot_does_not_set_the_speed_of_nonstop_talk():
    # Sixty lines with no real pause; only lines 30-32 have too much to say.
    segs = [Segment(i * 3.2, i * 3.2 + 3, "x") for i in range(60)]
    natural = [2.6] * 60
    natural[30:33] = [4.4, 4.4, 4.4]
    placed = dub.plan(segs, natural, 200, 1.6)
    speeds = [p.speed for p in placed]
    assert max(speeds[30:33]) > 1.2
    assert speeds[0] == dub.CALM_SPEED and speeds[-1] == dub.CALM_SPEED
    assert sum(s > 1.0 for s in speeds) < 15  # the tight lines and those easing in and out
    assert all(abs(a - b) <= dub.MAX_STEP + 1e-9 for a, b in zip(speeds, speeds[1:]))
    assert all(-1e-9 <= p.start - s.start <= dub.SYNC_LAG + 1e-9 for p, s in zip(placed, segs))


def test_a_respoken_line_that_is_still_too_long_is_stretched_to_fit(monkeypatch):
    class Stubborn:
        rate = 16000

        def chars_per_second(self, voice):
            return 20.0  # a guess twice too fast, so the line is spoken again

        def speak_many(self, lines):
            # Asked for more speed, this voice delivers only half of it.
            return [np.full(int(len(line.text) / 10 / (1 + (line.speed - 1) / 2) * self.rate),
                            0.1, dtype=np.float32) for line in lines]

    stretched = []
    monkeypatch.setattr(dub, "stretch",
                        lambda audio, tempo, rate: stretched.append(tempo) or audio[:int(len(audio) / tempo)])
    # 4 s of speech for a 2 s slot, with the next line straight after.
    segs = [Segment(0.0, 2.0, "х" * 40), Segment(2.1, 4.0, "х" * 10)]
    track, placed = dub.synthesize(segs, {0: dub.Voice("dmitri")}, 10, 1.6, Stubborn())
    assert stretched and max(stretched) > 1.02  # not left overlapping the next line


def test_piper_is_asked_for_the_speed_it_really_delivers():
    piper = dub.Piper()
    assert piper.length_scale(1.0) == 1.0
    # Measured: a setting of 0.7 speaks 1.26x faster, not the 1.43x its number says.
    assert piper.length_scale(1.26) == pytest.approx(0.7)
    assert piper.length_scale(dub.CALM_SPEED) > 1 / dub.CALM_SPEED
    scales = [piper.length_scale(s / 100) for s in range(85, 200, 5)]
    assert scales == sorted(scales, reverse=True) and min(scales) >= 0.4


def test_a_line_does_not_talk_on_into_the_pause_after_it():
    # 3.2 s to say in a 3 s slot, then a 4 s pause: without a limit the calm
    # pace would stretch it to 3.5 s, half a second after the speaker stopped.
    segs = [Segment(0, 3, "a"), Segment(7, 9, "b")]
    placed = dub.plan(segs, [3.2, 1.0], 20, 1.6)
    assert placed[0].start + placed[0].length <= 3 + dub.SYNC_OVERRUN + 1e-6
    assert placed[0].speed < 1.0  # it still takes the time it is allowed
    assert placed[1].speed == dub.CALM_SPEED


def test_a_line_that_cannot_finish_in_time_is_spoken_at_top_speed():
    placed = dub.plan([Segment(0, 2, "a"), Segment(9, 11, "b")], [6.0, 1.0], 20, 1.6)
    assert placed[0].speed == 1.6 and placed[1].start == 9


def test_hours_of_nonstop_talk_are_planned_quickly():
    import time

    segs = [Segment(i * 3.0, i * 3.0 + 2.95, "x") for i in range(2000)]
    began = time.perf_counter()
    placed = dub.plan(segs, [3.6] * 2000, 6000, 1.6)
    assert time.perf_counter() - began < 3  # was 5 s before long talk was planned in stretches
    assert all(-1e-9 <= p.start - s.start <= dub.SYNC_LAG + 1e-6 for p, s in zip(placed, segs))
    speeds = [p.speed for p in placed]
    assert all(abs(a - b) <= dub.MAX_STEP + 1e-9 for a, b in zip(speeds, speeds[1:]))


def test_a_pitch_shifted_voice_is_guessed_at_its_real_length():
    piper = dub.Piper()
    # Lowering the pitch by resampling makes a line longer, so fewer characters a second.
    assert piper.chars_per_second(dub.Voice("denis", -36)) == pytest.approx(16.5 * 0.82)
    assert piper.chars_per_second(dub.Voice("denis")) == 16.5
