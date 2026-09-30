#!/usr/bin/env bash
# Apply patches/<name>.patch to src/<name> for every patch in this directory.
set -euo pipefail
ws="$(cd "$(dirname "$0")/.." && pwd)"
for patch in "$ws"/patches/*.patch; do
  name="$(basename "$patch" .patch)"
  echo "Applying $name"
  git -C "$ws/src/$name" apply --whitespace=nowarn "$patch"
done
