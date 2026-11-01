#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
output="${project_root}/content/tools/sandbox/seccomp_exec"
mkdir -p "$(dirname "${output}")"
temporary="${output}.tmp.$$"
trap 'rm -f "${temporary}"' EXIT
gcc -O2 -Wall -Wextra -Werror "${project_root}/src/evaluation/seccomp_exec.c" \
  -o "${temporary}"
chmod 0755 "${temporary}"
mv "${temporary}" "${output}"
trap - EXIT
sha256sum "${output}"
