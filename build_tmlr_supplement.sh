#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE_DIR="$(mktemp -d)"
ARCHIVE_TMP="$(mktemp -t complexity-deep-tmlr-supplement).zip"
trap 'rm -rf "${STAGE_DIR}" "${ARCHIVE_TMP}"' EXIT

rsync -a \
  --exclude='.DS_Store' \
  --exclude='.pytest_cache/' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='checkpoints/' \
  --exclude='runs/' \
  "${ROOT_DIR}/supplementary_code/" \
  "${STAGE_DIR}/supplementary_code/"

(
  cd "${STAGE_DIR}"
  zip -X -q -r "${ARCHIVE_TMP}" supplementary_code
)

mkdir -p "${ROOT_DIR}/dist"
cp "${ARCHIVE_TMP}" "${ROOT_DIR}/dist/complexity_deep_tmlr_supplement.zip"
cp "${ARCHIVE_TMP}" "${ROOT_DIR}/supplementary_code_tmlr.zip"

unzip -tq "${ROOT_DIR}/dist/complexity_deep_tmlr_supplement.zip"
echo "Built:"
shasum -a 256 \
  "${ROOT_DIR}/dist/complexity_deep_tmlr_supplement.zip" \
  "${ROOT_DIR}/supplementary_code_tmlr.zip"
