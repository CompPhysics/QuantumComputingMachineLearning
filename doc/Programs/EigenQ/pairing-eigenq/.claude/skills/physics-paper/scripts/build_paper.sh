#!/usr/bin/env bash
# Compile a LaTeX article twice; fail on errors or unresolved references.
set -euo pipefail
TEX="${1:?usage: build_paper.sh paper/eigenq_pairing.tex}"
DIR="$(dirname "$TEX")"; BASE="$(basename "$TEX" .tex)"
cd "$DIR"
pdflatex -interaction=nonstopmode -halt-on-error "$BASE.tex" >/tmp/tex1.log 2>&1
pdflatex -interaction=nonstopmode -halt-on-error "$BASE.tex" >/tmp/tex2.log 2>&1
if grep -qiE "Citation .* undefined|Reference .* undefined" /tmp/tex2.log; then
  echo "UNRESOLVED references/citations:"; grep -iE "undefined" /tmp/tex2.log; exit 1
fi
echo "built $DIR/$BASE.pdf"
