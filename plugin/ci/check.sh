#!/bin/bash
set -e -u -o pipefail

THIS_SCRIPT_DIR=$( cd "$( dirname "$0" )" ; pwd )
PROJECT_DIR=$( cd "$THIS_SCRIPT_DIR/.." ; pwd )

EMULATOR_PATH=/usr/local/google-cloud-sdk/platform/bigtable-emulator/cbtemulator
EMULATOR_PID=
EMULATOR_LOG=

onExit() {
  [ -z "$EMULATOR_PID" ] || kill "$EMULATOR_PID"
  [ -z "$EMULATOR_LOG" ] || rm -f "$EMULATOR_LOG"
}

log() {
  echo "$@" > /dev/stderr
}

fail() {
  log "$@"
  exit 1
}

trap onExit exit

log "Testing if the cc sources are properly formatted."

find \( -name \*.cpp -o -name \*.h \) -print0 \
  | xargs -0 clang-format-10 --dry-run --Werror

log "Compiling and installing the project"

cd "$PROJECT_DIR"
python3 setup.py install

log "Installing the project"

EMULATOR_LOG=$(mktemp)
"$EMULATOR_PATH" -host 127.0.0.1 -port 0 >"$EMULATOR_LOG" &
EMULATOR_PID=$!

# Wait until the CBT emulator spits out a line with:
# "Cloud Bigtable emulator running on 127.0.0.1:12345"
BIGTABLE_EMULATOR_HOST=$(
  timeout 30 sh -c "tail -f \"$EMULATOR_LOG\" || true" \
  | grep -im 1 "running on" \
  | fmt -w 1 \
  | grep '^[0-9]\+\.[0-9]\+\.[0-9]\+\.[0-9]\+:[0-9]\+$'
  )

[ -n "$BIGTABLE_EMULATOR_HOST" ] || \
  fail "Failed to find the emulator port number"
export BIGTABLE_EMULATOR_HOST

log "Cloud Bigtable emulator is running on port $BIGTABLE_EMULATOR_HOST."
log "Testing the project."

# Create a dumy table.
cbt -project fake_project -instance fake_instance createtable fake_table
# Read something from it.
LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/torch/lib \
  python3 \
  pytorch_bigtable/example.py

log "Successfully ran CBT pyTorch example."
