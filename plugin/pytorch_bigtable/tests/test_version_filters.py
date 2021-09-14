from datetime import datetime
from unittest import TestCase
from pytorch_bigtable import version_filters as filters


class VersionFilterTest(TestCase):

  def test_latest(self):
    expected = 'cells_per_column_limit_filter: 1\n'
    self.assertEqual(expected, repr(filters.latest()))

  def test_timestamp_range_micros(self):
    start_timestamp = int(datetime(2020, 10, 10, 12, 0, 0).timestamp())
    end_timestamp = int(datetime(2100, 10, 10, 13, 0, 0).timestamp())
    expected = ('timestamp_range_filter {\n'
                f'  start_timestamp_micros: {start_timestamp}\n'
                f'  end_timestamp_micros: {end_timestamp}\n'
                '}\n')
    self.assertEqual(expected, repr(
      filters.timestamp_range_micros(start_timestamp, end_timestamp)))
