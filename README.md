# vtranslate

Downloads a video (or uses a file on your disk) and dubs it into Russian, in the
style of Yandex's video translation: Russian voices on top, with the original
turned down but still audible. Each speaker gets their own male or female voice.

Translation is done by DeepL, which reads the surrounding conversation and the
video's title, so idioms and references are translated by meaning. Everything
else (speech recognition, speaker detection, voices, mixing) runs on your own
computer.

## Install

On Linux, paste this into a terminal. Nothing else needs installing first:

```bash
curl -LsSf https://raw.githubusercontent.com/snysik1337/vtranslate/main/install.sh | sh
```

Then open a new terminal. It takes about 750 MB of disk space, and running the
same command again updates it. The first dub downloads the speech models (about
170 MB) and about 60 MB for each Russian voice it uses, only once.

To uninstall: `uv tool uninstall vtranslate`

## DeepL key

Each person needs their own free DeepL API key:

1. Sign up for **DeepL API Free** at https://www.deepl.com/pro-api.
2. Copy your key from the API Keys page of your DeepL account.
3. Run vtranslate. The first time, it asks for the key and saves it for your
   user only (`~/.config/vtranslate/deepl-key`). To change it later:

```bash
vtranslate --set-deepl-key
```

The free plan includes a monthly character allowance (500,000 in DeepL's
documentation, though some accounts show 1,000,000), which is roughly 12 to 14
hours of talking-heavy video per 500,000. It refills every month and the same
key keeps working. Each run shows how much is used. The conversation DeepL reads as
context is not counted, and a translated video is saved, so rerunning it costs
nothing.

If DeepL can't be used (allowance used up, no internet, a language DeepL
doesn't translate), vtranslate stops with a message saying why. The key and
allowance are checked before any slow work starts, and a rerun continues from
the last finished step.

## Use

```bash
vtranslate 'https://www.youtube.com/watch?v=...'
```

```bash
vtranslate lecture.mp4 interview.mkv
```

Finished videos go to `~/Videos/vtranslate`. Video is copied, not re-encoded, so
picture quality is unchanged. If a run is interrupted, run the same command again:
it continues from the last finished step.

| Option | What it does |
|---|---|
| `-o FOLDER` | Where to save finished videos |
| `-q 720` | Maximum download height (default 1080) |
| `-m large-v3-turbo` | More accurate speech recognition for accents and noise, about 7x slower (default `base`; `small` is in between) |
| `--voice male` / `female` | One voice for everything, skips speaker detection |
| `--speakers 3` | Exact speaker count, if you know it. The best fix for wrong voices |
| `--speaker-threshold 0.65` | Lower finds more speakers, higher finds fewer |
| `--best` | Use Microsoft's online voices when they work (nicer, slower) |
| `--tts edge` | Online Microsoft voices only. `piper` is offline (default), `auto` tries Microsoft first |
| `--from en` | Set the spoken language if detection gets it wrong |
| `--original-volume 0.1` | Volume of the original during speech (default 0.15) |
| `--max-speed 1.4` | Fastest allowed dub speech (default 1.6) |
| `--cookies-from-browser firefox` | Use your login for sites that require one |
| `--set-deepl-key` | Save a new DeepL key |

## How it works

| Step | Tool | Runs |
|---|---|---|
| Download | yt-dlp, sound first and picture in the background | online |
| Speech recognition | faster-whisper `base` | local |
| Speaker detection | one sherpa-onnx CAM++ voiceprint per sentence | local |
| Translation | DeepL, with the surrounding lines and the title as context | online |
| Voices | Piper (`--best` tries Microsoft's voices first) | local |
| Mix | PyAV, which contains FFmpeg's libraries, so no ffmpeg program is needed | local |

Lines separated by short pauses are spoken as one phrase at one steady speed,
the slowest that keeps every line within half a second of the original. When
Russian runs longer than the original, the whole phrase speeds up a little
instead of single lines rushing.

Each run prints how long every stage took.

## Things to know

- Speech recognition and translation still make mistakes, especially with names.
- Russian verb gender can be wrong, because DeepL does not know who is speaking.
- Speaker gender is guessed from voice pitch. Sentences that sound like the same
  person get one voice, and a few seconds of speech is too little to tell someone
  apart, so a person who only says a word or two gets the closest voice.
  `--speakers N` sets the number of speakers yourself.
- Whisper sometimes skips quiet speech. Gaps of 2.5 seconds or more are read
  again, but shorter skipped lines stay missing from the dub.
- Microsoft's free voices are unofficial and can fail. If they do, the whole video
  uses Piper, so voices never change partway through.
- Only videos are supported: an address or file with sound but no picture is
  refused straight away. For an address holding a list of videos, the first one
  is used.
- If the Russian runs on past the end of the video, the final words are kept and
  the picture holds its last frame until they finish.

## Tests

```bash
.venv/bin/python -m pytest tests
```
