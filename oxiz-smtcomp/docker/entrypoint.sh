#!/usr/bin/env bash
# Entrypoint for oxiz-smtcomp runner Docker image.
# Forwards all arguments to the smtcomp2026 binary. When invoked with
# no arguments, prints --help so the user discovers the interface.
#
# Exit codes match the binary itself:
#   0  = sat
#   10 = unsat
#   20 = unknown / timeout / error

set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  exec smtcomp2026 --help
fi

exec smtcomp2026 "$@"
