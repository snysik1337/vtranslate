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
    echo "Installing uv, which sets up vtranslate's Python for it..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    PATH="$HOME/.local/bin:$PATH"
  fi
fi

# The computer's own Python is used when it is a version every part supports;
# otherwise uv downloads one (about 110 MB more).
echo "Installing vtranslate (about 650 MB of disk space)..."
uv tool install --quiet --python-preference system --python ">=3.11,<3.15" \
  --force --reinstall-package vtranslate "$source"
uv tool update-shell >/dev/null 2>&1 || true

echo
echo "Done. Open a new terminal and run:  vtranslate 'https://www.youtube.com/watch?v=...'"
echo "The first run asks for your free DeepL key. To remove: uv tool uninstall vtranslate"
