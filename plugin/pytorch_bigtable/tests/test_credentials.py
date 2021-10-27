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

# disable module docstring for tests
# pylint: disable=C0114
# disable class docstring for tests
# pylint: disable=C0115
# disable warning for access to protected members
# pylint: disable=W0212
from unittest import TestCase
from tempfile import NamedTemporaryFile
from pytorch_bigtable.bigtable_dataset import ServiceAccountJson


class ServiceAccountJsonTest(TestCase):
  def test_reading_from_file(self):
    with NamedTemporaryFile(buffering=0) as tmpfile:
      tmpfile.write("example_content".encode())
      json_creds = ServiceAccountJson.read_from_file(tmpfile.name)
      self.assertEqual("example_content", json_creds._json_text)
