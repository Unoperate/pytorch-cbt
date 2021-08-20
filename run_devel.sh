#!/bin/bash
set -e -u -o pipefail

ROOT_DIR=$( cd "$( dirname "$0" )" ; pwd )
cd "$ROOT_DIR"

docker build -t pythorch_cbt_devel .
docker run --rm -it --volume "$ROOT_DIR/plugin:/opt/pytorch" pythorch_cbt_devel
