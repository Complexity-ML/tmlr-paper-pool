#!/usr/bin/env bash
set -euo pipefail

NATURE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${NATURE_DIR}/.." && pwd)"
BUILD_ROOT="${NATURE_DIR}/output/build"
PDF_DIR="${NATURE_DIR}/output/pdf"
DIST_DIR="${REPO_ROOT}/dist"

PDFLATEX="${PDFLATEX:-$(command -v pdflatex || true)}"
BIBTEX="${BIBTEX:-$(command -v bibtex || true)}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"

if [[ -z "${PDFLATEX}" && -x /Library/TeX/texbin/pdflatex ]]; then
  PDFLATEX=/Library/TeX/texbin/pdflatex
fi
if [[ -z "${BIBTEX}" && -x /Library/TeX/texbin/bibtex ]]; then
  BIBTEX=/Library/TeX/texbin/bibtex
fi
for tool in PDFLATEX BIBTEX PYTHON_BIN; do
  if [[ -z "${!tool}" ]]; then
    echo "missing required tool: ${tool}" >&2
    exit 1
  fi
done

if grep -Eq '\[AUTHOR ACTION REQUIRED|\[NAME\]|\[EMAIL\]|\[INSTITUTION\]|\[POSTAL ADDRESS\]' "${NATURE_DIR}/cover_letter.tex"; then
  echo "cover_letter.tex still contains author placeholders" >&2
  exit 1
fi

mkdir -p "${BUILD_ROOT}/manuscript" "${BUILD_ROOT}/supplement" \
  "${BUILD_ROOT}/cover" "${BUILD_ROOT}/matplotlib" "${PDF_DIR}" "${DIST_DIR}"

MPLCONFIGDIR="${BUILD_ROOT}/matplotlib" \
  "${PYTHON_BIN}" "${REPO_ROOT}/supplementary_code/scripts/render_verified_architecture.py"
MPLCONFIGDIR="${BUILD_ROOT}/matplotlib" \
  "${PYTHON_BIN}" "${REPO_ROOT}/supplementary_code/scripts/render_300m_scaling_figure.py"

cd "${NATURE_DIR}"
"${PDFLATEX}" -interaction=nonstopmode -halt-on-error \
  -output-directory="${BUILD_ROOT}/manuscript" manuscript.tex
(
  cd "${BUILD_ROOT}/manuscript"
  BIBINPUTS="${NATURE_DIR}:" "${BIBTEX}" manuscript
)
"${PDFLATEX}" -interaction=nonstopmode -halt-on-error \
  -output-directory="${BUILD_ROOT}/manuscript" manuscript.tex
"${PDFLATEX}" -interaction=nonstopmode -halt-on-error \
  -output-directory="${BUILD_ROOT}/manuscript" manuscript.tex

"${PDFLATEX}" -interaction=nonstopmode -halt-on-error \
  -output-directory="${BUILD_ROOT}/supplement" supplementary_information.tex
"${PDFLATEX}" -interaction=nonstopmode -halt-on-error \
  -output-directory="${BUILD_ROOT}/cover" cover_letter.tex

for log in \
  "${BUILD_ROOT}/manuscript/manuscript.log" \
  "${BUILD_ROOT}/supplement/supplementary_information.log" \
  "${BUILD_ROOT}/cover/cover_letter.log"; do
  if grep -Eq 'LaTeX Warning|Overfull|Underfull|undefined' "${log}"; then
    echo "layout warning in ${log}" >&2
    grep -E 'LaTeX Warning|Overfull|Underfull|undefined' "${log}" >&2
    exit 1
  fi
done

cp "${BUILD_ROOT}/manuscript/manuscript.pdf" "${PDF_DIR}/manuscript.pdf"
cp "${BUILD_ROOT}/supplement/supplementary_information.pdf" \
  "${PDF_DIR}/supplementary_information.pdf"
cp "${BUILD_ROOT}/cover/cover_letter.pdf" "${PDF_DIR}/cover_letter.pdf"

SUBMISSION_BUNDLE="${DIST_DIR}/nature_machine_intelligence_submission_bundle.zip"
rm -f "${SUBMISSION_BUNDLE}"
(
  cd "${PDF_DIR}"
  zip -q "${SUBMISSION_BUNDLE}" manuscript.pdf \
    supplementary_information.pdf cover_letter.pdf
)

CODE_BUNDLE="${DIST_DIR}/nature_machine_intelligence_supplementary_code.zip"
STAGE_DIR="$(mktemp -d)"
trap 'rm -rf "${STAGE_DIR}"' EXIT
mkdir -p "${STAGE_DIR}/supplementary_code"
rsync -a \
  --exclude '.DS_Store' \
  --exclude '.pytest_cache' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude 'checkpoints' \
  --exclude 'runs' \
  --exclude 'figures' \
  "${REPO_ROOT}/supplementary_code/" "${STAGE_DIR}/supplementary_code/"
mkdir -p "${STAGE_DIR}/supplementary_code/figures"
cp "${REPO_ROOT}/supplementary_code/figures/architecture_complexity_deep.pdf" \
  "${STAGE_DIR}/supplementary_code/figures/"
cp "${REPO_ROOT}/supplementary_code/figures/architecture_complexity_deep.png" \
  "${STAGE_DIR}/supplementary_code/figures/"
cp "${REPO_ROOT}/supplementary_code/figures/fig_300m_loss_curves.pdf" \
  "${STAGE_DIR}/supplementary_code/figures/"
cp "${REPO_ROOT}/supplementary_code/figures/fig_300m_loss_curves.png" \
  "${STAGE_DIR}/supplementary_code/figures/"
rm -f "${CODE_BUNDLE}"
(
  cd "${STAGE_DIR}"
  zip -qr "${CODE_BUNDLE}" supplementary_code
)

echo "Built:"
ls -lh "${PDF_DIR}/manuscript.pdf" \
  "${PDF_DIR}/supplementary_information.pdf" \
  "${PDF_DIR}/cover_letter.pdf" \
  "${SUBMISSION_BUNDLE}" "${CODE_BUNDLE}"
