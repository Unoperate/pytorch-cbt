#!/bin/bash
set -e -u -o pipefail

THIS_SCRIPT_DIR=$( cd "$( dirname "$0" )" ; pwd )
PROJECT_DIR=$( cd "$THIS_SCRIPT_DIR/.." ; pwd )

log() {
  echo "$@" > /dev/stderr
}

fail() {
  log "$@"
  exit 1
}

log "Testing if the cc sources are properly formatted."

find \( -name \*.cc -o -name \*.h \) -print0 \
  | xargs -0 clang-format-10 --dry-run --Werror

log "Compiling the project"
cd "$PROJECT_DIR"
python3 setup.py develop

log "Testing the project"
python3 setup.py test

log "Running clang-tidy"
clang-tidy-10 $(find -name \*.cc -o -name \*.h) \
  -- \
  -std=c++17 \
  -I /usr/local/lib/python3.8/dist-packages/torch/include \
  -I /usr/local/lib/python3.8/dist-packages/torch/include/torch/csrc/api/include \
  -I /usr/include/python3.8
