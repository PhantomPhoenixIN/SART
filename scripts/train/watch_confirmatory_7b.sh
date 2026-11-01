#!/usr/bin/env bash
# Passive project heartbeat. It never submits, cancels, or changes PBS jobs.
# Stops only after the project completion marker is written.
set -u
project_root="/home/swaminathanj/research_artifacts/SART"
cd "$project_root"
log="logs/confirmatory_7b/monitor_until_complete.log"
pid_file="logs/confirmatory_7b/monitor_until_complete.pid"
complete_marker="experiments/PROJECT_COMPLETE"
interval_seconds="${SART_MONITOR_INTERVAL_SECONDS:-300}"
mkdir -p "$(dirname "$log")"
printf '%s\n' "$$" > "$pid_file"
if [[ ! "$interval_seconds" =~ ^[0-9]+$ ]] || (( interval_seconds < 60 )); then
  echo "SART_MONITOR_INTERVAL_SECONDS must be an integer >= 60" >&2
  exit 2
fi
cycle=0
printf '%s watcher started (interval=%ss; stop marker=%s)\n' \
  "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$interval_seconds" "$complete_marker" >> "$log"
while [[ ! -e "$complete_marker" ]]; do
  {
    printf '\n===== %s cycle=%06d =====\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$cycle"
    env/bin/python scripts/train/report_pbs_status.py 2>&1 || true
    qstat -u "$USER" 2>&1 | awk 'NR==1 || NR==2 || /gpuq/ {print}'
  } >> "$log"
  cycle=$((cycle + 1))
  sleep "$interval_seconds"
done
printf '%s completion marker found; watcher stopping\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" >> "$log"
rm -f "$pid_file"
