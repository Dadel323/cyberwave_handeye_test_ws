#!/usr/bin/env bash
# Regenerate patches/<name>.patch from the working tree of src/<name>.
# Run after changing a third-party package; untracked files are included.
set -euo pipefail
ws="$(cd "$(dirname "$0")/.." && pwd)"
for patch in "$ws"/patches/*.patch; do
  name="$(basename "$patch" .patch)"
  repo="$ws/src/$name"
  git -C "$repo" add -N .
  git -C "$repo" diff --binary > "$patch"
  git -C "$repo" reset -q
  echo "Refreshed $name"
done
