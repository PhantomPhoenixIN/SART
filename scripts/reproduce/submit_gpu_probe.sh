#!/usr/bin/env bash
set -euo pipefail
project_root="/home/swaminathanj/research_artifacts/SART"
stamp="$(date -u '+%Y%m%dT%H%M%SZ')"
qsub \
  -o "${project_root}/logs/gpu_probe_${stamp}.pbs.stdout" \
  -e "${project_root}/logs/gpu_probe_${stamp}.pbs.stderr" \
  "${project_root}/run_on_astra/gpu_probe.pbs"
