#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${SCRIPT_DIR}/tmp/build"
PDF_DIR="${SCRIPT_DIR}/output/pdf"
DIST_DIR="${SCRIPT_DIR}/dist"

mkdir -p "${BUILD_DIR}" "${PDF_DIR}" "${DIST_DIR}"
find "${BUILD_DIR}" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
mkdir -p "${BUILD_DIR}/figures"

cp "${SCRIPT_DIR}/main.tex" "${BUILD_DIR}/main.tex"
cp "${REPO_DIR}/references.bib" "${BUILD_DIR}/references.bib"
cp \
  "${SCRIPT_DIR}/figures/architecture_complexity_deep.pdf" \
  "${BUILD_DIR}/figures/architecture_complexity_deep.pdf"

(
  cd "${BUILD_DIR}"
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
)

cp "${BUILD_DIR}/main.pdf" \
  "${PDF_DIR}/token_identity_routing_arxiv.pdf"

(
  cd "${BUILD_DIR}"
  zip -q -r "${DIST_DIR}/token_identity_routing_arxiv_source.zip" \
    main.tex references.bib figures
)

echo "PDF: ${PDF_DIR}/token_identity_routing_arxiv.pdf"
echo "Source: ${DIST_DIR}/token_identity_routing_arxiv_source.zip"
