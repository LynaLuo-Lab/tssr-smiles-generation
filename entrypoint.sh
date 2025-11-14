#!/usr/bin/env bash
set -euo pipefail

SCRIPT="python RunScript.py"

show_help() {
  cat <<'EOF'
Usage:
  docker run --rm -it --gpus all ghcr.io/lynaluo-lab/tssr-smiles-generation:latest <profile-name>
  docker run --rm -it --gpus all ghcr.io/lynaluo-lab/tssr-smiles-generation:latest help
  # Or pass raw args to the script:
  docker run --rm -it --gpus all ghcr.io/lynaluo-lab/tssr-smiles-generation:latest -- --your --raw --args

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
      exec ${SCRIPT} --pure-rl --seed 1949262625
      ;;
    PRL-Run2)
      exec ${SCRIPT} --pure-rl --seed 2683294732
      ;;
    PRL-Run3)
      exec ${SCRIPT} --pure-rl --seed 1103657151
      ;;
    PRL-Run4)
      exec ${SCRIPT} --pure-rl --seed 3321047177
      ;;
    PRL-Run5)
      exec ${SCRIPT} --pure-rl --seed 4251184328
      ;;
    FRL-Run1)
      exec ${SCRIPT} --seed 3368427155
      ;;
    FRL-Run2)
      exec ${SCRIPT} --seed 190166649
      ;;
    FRL-Run3)
      exec ${SCRIPT} --seed 2485260846
      ;;
    FRL-Run4)
      exec ${SCRIPT} --seed 2120889288
      ;;
    FRL-Run5)
      exec ${SCRIPT} --seed 1410668516
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
