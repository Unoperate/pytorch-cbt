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

"""Module implementing basic functions for obtaining BigTable Filters
for version filtering.
"""
from . import pbt_C
from typing import Union
from datetime import datetime


def latest() -> pbt_C.Filter:
  """Create a filter passing only the latest version of
  column's value for each row.

  Returns:
    pbt_C.Filter: Filter passing only most recent version of a value.
  """
  return pbt_C.latest_version_filter(1)


def timestamp_range(start: Union[int, float, datetime],
                    end: Union[int, float, datetime]) -> pbt_C.Filter:
  """Create a filter passing all values which timestamp is
  from the specified range, exclusive at the start and inclusive at the end.

  Args:
    start: The start of the row range (inclusive). This can be either a python
    datetime or a number (int of float) representing seconds since epoch.
    end: The end of the row range (exclusive). Same as start, this can be a
    datetime or a number of seconds since epoch.
  Returns:
    pbt_C.Filter: Filter passing only values' versions from the specified range.
  """
  if isinstance(start, datetime):
    start_timestamp = int(start.timestamp() * 1e6)
  else:
    start_timestamp = int(start * 1e6)

  if isinstance(end, datetime):
    end_timestamp = int(end.timestamp() * 1e6)
  else:
    end_timestamp = int(end * 1e6)

  return pbt_C.timestamp_range_micros(start_timestamp, end_timestamp)


def timestamp_range_micros(start_timestamp: Union[int, float],
                           end_timestamp: Union[int, float]) -> pbt_C.Filter:
  """Create a filter passing all values which timestamp is
  from the specified range, exclusive at the start and inclusive at the end.

  Args:
    start_timestamp: The start of the row range (inclusive). It is a number (
    int or float) representing number of microseconds since epoch.
    end_timestamp: The end of the row range (exclusive).
  Returns:
    pbt_C.Filter: Filter passing only values' versions from the specified range.
  """
  return pbt_C.timestamp_range_micros(int(start_timestamp), int(end_timestamp))
