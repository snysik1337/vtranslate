#!/bin/sh
# Installs (or updates) vtranslate. Nothing else needs to be installed first:
#
#   curl -LsSf https://raw.githubusercontent.com/snysik1337/vtranslate/main/install.sh | sh
#
# Run from a downloaded copy of the project (./install.sh), it installs that copy.
set -eu

ARCHIVE="https://github.com/snysik1337/vtranslate/archive/refs/heads/main.tar.gz"
here=$(cd "$(dirname -- "$0")" 2>/dev/null && pwd || true)
if [ -n "$here" ] && grep -qs '^name = "vtranslate"' "$here/pyproject.toml"; then
  source="$here"
else
  source="$ARCHIVE"
fi

# uv installs Python packages, and Python itself if the computer's does not suit.
if ! command -v uv >/dev/null 2>&1; then
  if [ -x "$HOME/.local/bin/uv" ]; then
    PATH="$HOME/.local/bin:$PATH"
  else
    echo "Installing uv, which installs vtranslate and its parts..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    PATH="$HOME/.local/bin:$PATH"
  fi
fi

# The computer's own Python is used when it is a version every part supports;
# otherwise uv downloads one (about 110 MB more).
echo "Installing vtranslate (about 610 MB of disk space)..."
# Everything by default: speaker detection, online voices and YouTube support.
uv tool install --quiet --python-preference system --python ">=3.11,<3.15" \
  --force --reinstall-package vtranslate "vtranslate[speakers,online,youtube] @ $source"

# Other projects' packages bring files a Russian dub never opens, about 37 MB:
# the models of Piper's Hebrew and Arabic helpers, and a download accelerator
# that the model download works without. Piper's pronunciation data for other
# languages stays: it reads names a translation leaves in Greek, Korean or
# other letters, which would be skipped without it.
"$(uv tool dir)/vtranslate/bin/python" - <<'TRIM' || true
import importlib.util
import pathlib
import shutil


def folder(package):
    spec = importlib.util.find_spec(package)
    return pathlib.Path(spec.origin).parent if spec and spec.origin else None


piper = folder("piper")
if piper:
    (piper / "hebrew" / "nakdimon.onnx").unlink(missing_ok=True)
    (piper / "tashkeel" / "model.onnx").unlink(missing_ok=True)
xet = folder("hf_xet")
if xet:
    shutil.rmtree(xet, ignore_errors=True)
    for record in xet.parent.glob("hf_xet-*.dist-info"):
        shutil.rmtree(record, ignore_errors=True)
TRIM
uv tool update-shell >/dev/null 2>&1 || true

echo
echo "Done. Open a new terminal and run:  vtranslate 'https://www.youtube.com/watch?v=...'"
echo "The first run asks for your free DeepL key. To remove: uv tool uninstall vtranslate"
