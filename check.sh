#!/usr/bin/env bash
# Run the acceptance tests for one drill (or all of them).
#   ./check.sh m01            # test your mine/m01_*.py
#   ./check.sh p07 -x         # extra args go to pytest
#   ./check.sh all            # every drill
#   ./check.sh m01 --solutions   # run the reference solution instead
set -euo pipefail
cd "$(dirname "$0")"

id="${1:-all}"
shift || true
args=()
for a in "$@"; do
  if [[ "$a" == "--solutions" ]]; then export DRILL_IMPL=solutions; else args+=("$a"); fi
done

if [[ "$id" == "all" ]]; then
  targets=(tests)
else
  shopt -s nullglob
  targets=(tests/test_"${id}"_*.py)
  if (( ${#targets[@]} == 0 )); then
    echo "no test file matches tests/test_${id}_*.py" >&2
    exit 1
  fi
fi

exec .venv/bin/pytest -q "${targets[@]}" "${args[@]}"
