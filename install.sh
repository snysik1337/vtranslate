#!/usr/bin/env bash
# Installs the `vtranslate` command into ~/.local/bin.
set -euo pipefail
cd "$(dirname "$0")"

if command -v uv >/dev/null; then
  uv=uv
else
  # uv fetches a compatible Python: the speech libraries do not support 3.14 yet.
  boot="$(mktemp -d)"
  trap 'rm -rf "$boot"' EXIT
  python3 -m venv "$boot"
  "$boot/bin/pip" install -q uv
  uv="$boot/bin/uv"
fi

"$uv" tool install --python 3.12 --force .
echo
echo "Installed. Try: vtranslate --help"
echo "If the command is not found, add ~/.local/bin to your PATH."
