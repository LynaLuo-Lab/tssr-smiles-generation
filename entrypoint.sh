#!/usr/bin/env bash
set -euo pipefail

SCRIPT="python RunScript.py"

show_help() {
  cat <<'EOF'
Usage:
  docker run --rm -it --gpus all tssr-smiles:latest <profile-name>
  docker run --rm -it --gpus all tssr-smiles:latest help
  # Or pass raw args to the script:
  docker run --rm -it --gpus all tssr-smiles:latest -- --your --raw --args

Available profiles:
  PRL-Run1
  PRL-Run2
  PRL-Run3
  PRL-Run4
  PRL-Run5

  FRL-Run1
  FRL-Run2
  FRL-Run3
  FRL-Run4
  FRL-Run5
EOF
}

run_profile() {
  case "${1:-help}" in
    PRL-Run1)
      exec ${SCRIPT} --pure-rl --seed 640011233
      ;;
    PRL-Run2)
      exec ${SCRIPT} --pure-rl --seed 1789382160
      ;;
    PRL-Run3)
      exec ${SCRIPT} --pure-rl --seed 3478580130
      ;;
    PRL-Run4)
      exec ${SCRIPT} --pure-rl --seed 3015646651
      ;;
    PRL-Run5)
      exec ${SCRIPT} --pure-rl --seed 1476376261
      ;;
    FRL-Run1)
      exec ${SCRIPT} --seed 1999133639
      ;;
    FRL-Run2)
      exec ${SCRIPT} --seed 1527591437
      ;;
    FRL-Run3)
      exec ${SCRIPT} --seed 1290877492
      ;;
    FRL-Run4)
      exec ${SCRIPT} --seed 3923673192
      ;;
    FRL-Run5)
      exec ${SCRIPT} --seed 1900098291
      ;;
    help|--help|-h|"")
      show_help; exit 0
      ;;
    --)
      shift; exec ${SCRIPT} "$@"
      ;;
    *)
      if [[ "${1}" == -* ]]; then
        exec ${SCRIPT} "$@"
      else
        echo "Unknown profile: ${1}" >&2
        echo; show_help; exit 1
      fi
      ;;
  esac
}

run_profile "${1:-help}"
