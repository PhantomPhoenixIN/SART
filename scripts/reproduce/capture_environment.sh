#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [[ $# -ne 1 ]]; then
  echo "usage: $0 EXPERIMENT_ID" >&2
  exit 2
fi
experiment_id="$1"
output_dir="${SART_ENVIRONMENT_DIR:-${project_root}/experiments/${experiment_id}/environment}"
if [[ "${output_dir}" != /* ]]; then
  output_dir="${project_root}/${output_dir}"
fi
mkdir -p "${output_dir}"

capture_optional() {
  local command_name="$1"
  local output_name="$2"
  if command -v "${command_name}" >/dev/null 2>&1; then
    "${command_name}" >"${output_dir}/${output_name}" 2>&1 || true
  else
    printf 'not found: %s\n' "${command_name}" >"${output_dir}/${output_name}"
  fi
}

{
  date -u '+%Y-%m-%dT%H:%M:%SZ'
  hostname
  pwd
  python --version 2>&1
  python -c 'import sys; print(sys.executable)'
  python -m pip freeze
  printf '\nPBS environment:\n'
  env | sort | awk -F= '/^PBS_/ || /^CUDA_VISIBLE_DEVICES=/ || /^SART_NUM_WORKERS=/ {print $1 "=" $2}'
} >"${output_dir}/environment.txt.tmp"
mv "${output_dir}/environment.txt.tmp" "${output_dir}/environment.txt"

capture_optional nvidia-smi nvidia-smi.txt
capture_optional nvcc nvcc.txt
capture_optional gcc gcc.txt
capture_optional g++ gxx.txt
capture_optional javac javac.txt
capture_optional qstat qstat-version.txt
if command -v git >/dev/null 2>&1; then
  git -C "${project_root}" rev-parse HEAD >"${output_dir}/git-commit.txt" 2>&1 || printf 'uncommitted-or-unavailable\n' >"${output_dir}/git-commit.txt"
  git -C "${project_root}" status --short >"${output_dir}/git-status.txt" 2>&1 || true
fi
