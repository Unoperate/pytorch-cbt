"""Module implementing basic functions for obtaining BigTable Filters
for version filtering.
"""
import pbt_C
from typing import Union
from datetime import datetime


def latest() -> pbt_C.Filter:
  """Create a filter passing only the latest version of
  column's value for each row.

  Returns:
    pbt_C.Filter: Filter passing only most recent version of a value.
  """
  return pbt_C.latest_version_filter(1)


def timestamp_range_micros(start: Union[int, datetime],
                           end: Union[int, datetime]) -> pbt_C.Filter:
  """Create a filter passing all values which timestamp is
  from the specified range, exclusive at the start and inclusive at the end.

  Args:
    start: The start of the row range (inclusive).
    end: The end of the row range (exclusive).
  Returns:
    pbt_C.Filter: Filter passing only values' versions from the specified range.
  """
  start_timestamp = start if isinstance(start, int) else int(start.timestamp())
  end_timestamp = end if isinstance(end, int) else int(end.timestamp())
  return pbt_C.timestamp_range_micros(start_timestamp, end_timestamp)
