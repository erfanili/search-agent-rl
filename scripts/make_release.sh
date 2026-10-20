#!/usr/bin/env bash
# Export a public release of this repo: a snapshot of HEAD without the files marked export-ignore in
# .gitattributes (private notes), as a fresh git repo with one commit and none of this repo's history.
# Push that repo to the public remote; never make this repo itself public, since its history has the private files.
#   bash scripts/make_release.sh /data/erfan/search-agent-rl-data/release/search-agent-rl
set -euo pipefail

cd "$(dirname "$0")/.."
OUT=${1:?usage: make_release.sh <empty output dir>}
[[ -z $(git status --porcelain) ]] || { echo "commit or stash changes first: the release is built from HEAD"; exit 1; }
[[ ! -e $OUT || -z $(ls -A "$OUT") ]] || { echo "$OUT is not empty"; exit 1; }

mkdir -p "$OUT"
git archive --format=tar HEAD | tar -x -C "$OUT"
for f in $(git check-attr export-ignore --all -- $(git ls-files) | awk -F': ' '$3=="set"{print $1}'); do
    [[ ! -e $OUT/$f ]] || { echo "export-ignore failed for $f"; exit 1; }
done
git -C "$OUT" init -q
git -C "$OUT" add -A
git -C "$OUT" commit -qm "Release from $(git rev-parse --short HEAD)"
echo "release at $OUT ($(git -C "$OUT" ls-files | wc -l) files); private files left out:"
git check-attr export-ignore --all -- $(git ls-files) | awk -F': ' '$3=="set"{print "  " $1}'
