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

from datetime import datetime
from unittest import TestCase
from pytorch_bigtable import version_filters as filters


class VersionFilterTest(TestCase):

  def test_latest(self):
    expected = 'cells_per_column_limit_filter: 1\n'
    self.assertEqual(expected, repr(filters.latest()))

  def test_timestamp_range_micros(self):
    start_timestamp = datetime(2020, 10, 10, 12, 0, 0).timestamp() * 1e6
    end_timestamp = datetime(2100, 10, 10, 13, 0, 0).timestamp() * 1e6
    expected = ('timestamp_range_filter {\n'
                f'  start_timestamp_micros: {int(start_timestamp)}\n'
                f'  end_timestamp_micros: {int(end_timestamp)}\n'
                '}\n')
    self.assertEqual(expected, repr(
      filters.timestamp_range_micros(start_timestamp, end_timestamp)))

  def test_timestamp_range(self):
    start = datetime(2020, 10, 10, 12, 0, 0)
    end = datetime(2100, 10, 10, 13, 0, 0)
    expected = ('timestamp_range_filter {\n'
                f'  start_timestamp_micros: {int(start.timestamp() * 1e6)}\n'
                f'  end_timestamp_micros: {int(end.timestamp() * 1e6)}\n'
                '}\n')

    self.assertEqual(expected, repr(filters.timestamp_range(start, end)))

  def test_timestamp_consistency(self):
    start = datetime(2020, 10, 10, 12, 0, 0)
    end = datetime(2100, 10, 10, 13, 0, 0)
    self.assertEqual(repr(filters.timestamp_range(start, end)), repr(
      filters.timestamp_range(start.timestamp(), end.timestamp())))

    self.assertEqual(repr(filters.timestamp_range(start, end)), repr(
      filters.timestamp_range_micros(start.timestamp() * 1e6,
                                     end.timestamp() * 1e6)))
