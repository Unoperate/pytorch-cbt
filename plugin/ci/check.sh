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

find \( -name \*.cc -o -name \*.h \) -print0 \
  | xargs -0 clang-format-10 --dry-run --Werror

cd "$PROJECT_DIR"

log "Installing the project"
python3 setup.py develop

log "Testing the project"
LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/torch/lib \
	python3 -m unittest -v

EMULATOR_LOG=$(mktemp)
"$EMULATOR_PATH" -host 127.0.0.1 -port 0 >"$EMULATOR_LOG" &
EMULATOR_PID=$!

log "Running clang-tidy"
clang-tidy-10 $(find -name \*.cc -o -name \*.h) \
  -- \
  -std=c++17 \
  -I /usr/local/lib/python3.8/dist-packages/torch/include \
  -I /usr/local/lib/python3.8/dist-packages/torch/include/torch/csrc/api/include \
  -I /usr/include/python3.8

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

# Set default CBT project and instance
echo project = fake_project > ~/.cbtrc
echo instance = fake_instance >> ~/.cbtrc
# Create a dumy table.
cbt createtable fake_table
cbt createfamily fake_table cf1
# Read something from it.
LD_LIBRARY_PATH=/usr/local/lib/python3.8/dist-packages/torch/lib \
  python3 \
  pytorch_bigtable/example.py

log "Successfully ran CBT pyTorch example."
