#!/bin/bash

# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e -u -o pipefail

ROOT_DIR=$( cd "$( dirname "$0" )" ; pwd )
cd "$ROOT_DIR"
OUTPUT_DIR=$ROOT_DIR/dist

for python_path in $(find /opt/python -maxdepth 1 -name "cp3[6789]*")
do
  "$python_path/bin/pip" install torch==1.9.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
  "$python_path/bin/python" setup.py bdist_wheel -d "$OUTPUT_DIR"
done

for wheel_file in "$OUTPUT_DIR/"*
do
  mv "$wheel_file" "${wheel_file/linux/manylinux2014}"
done

