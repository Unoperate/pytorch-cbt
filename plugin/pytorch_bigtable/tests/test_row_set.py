from unittest import TestCase
from pytorch_bigtable import row_set
from pytorch_bigtable import row_range

class RowRangeTest(TestCase):
    def test_infinite(self):
        self.assertEqual("", repr(row_range.infinite()))

    def test_starting_at(self):
        expected = 'start_key_closed: "row1"\n'
        self.assertEqual(expected, repr(row_range.starting_at("row1")))

    def test_ending_at(self):
        expected = 'end_key_closed: "row1"\n'
        self.assertEqual(expected, repr(row_range.ending_at("row1")))

    def test_empty(self):
        expected = (
                'start_key_open: ""\n' +
                'end_key_open: "\\000"\n')

        self.assertEqual(expected, repr(row_range.empty()))

    def test_prefix(self):
        expected = (
                'start_key_closed: "row1"\n'
                'end_key_open: "row2"\n'
                )
        self.assertEqual(expected, repr(row_range.prefix("row1")))

    def test_right_open(self):
        expected = (
                'start_key_closed: "row1"\n'
                'end_key_open: "row2"\n'
                )
        self.assertEqual(expected, repr(row_range.right_open("row1", "row2")))

    def test_left_open(self):
        expected = (
                'start_key_open: "row1"\n'
                'end_key_closed: "row2"\n'
                )
        self.assertEqual(expected, repr(row_range.left_open("row1", "row2")))

    def test_open(self):
        expected = (
                'start_key_open: "row1"\n'
                'end_key_open: "row2"\n'
                )
        self.assertEqual(expected, repr(row_range.open_range("row1", "row2")))

    def test_closed(self):
        expected = (
                'start_key_closed: "row1"\n' +
                'end_key_closed: "row2"\n')
        self.assertEqual(expected, repr(row_range.closed_range("row1", "row2")))

class TestRowSet(TestCase):
    def test_empty(self):
        expected = ''
        self.assertEqual(expected, repr(row_set.empty()))

    def test_append_row(self):
        r_set = row_set.empty()
        r_set.append_row("row1")
        expected = 'row_keys: "row1"\n'
        self.assertEqual(expected, repr(r_set))

    def test_append_row_range(self):
        r_set = row_set.empty()
        r_set.append_range(row_range.closed_range("row1", "row2"))
        expected = (
              'row_ranges {\n' +
              '  start_key_closed: "row1"\n' +
              '  end_key_closed: "row2"\n' +
              '}\n')
        self.assertEqual(expected, repr(r_set))

    def test_from_rows_or_ranges(self):
        expected = (
                'row_keys: "row3"\n' +
                'row_keys: "row6"\n' +
                'row_ranges {\n' +
                '  start_key_closed: "row1"\n' +
                '  end_key_closed: "row2"\n' +
                '}\n' +
                'row_ranges {\n' +
                '  start_key_open: "row4"\n' +
                '  end_key_open: "row5"\n' +
                '}\n')

        r_set = row_set.from_rows_or_ranges(
                row_range.closed_range("row1", "row2"),
                "row3",
                row_range.open_range("row4", "row5"),
                "row6")
        self.assertEqual(expected, repr(r_set))

    def test_intersect(self):
        r_set = row_set.from_rows_or_ranges(row_range.open_range("row1", "row5"))
        r_set = r_set.intersect(row_range.closed_range("row3", "row7"))
        expected = (
                'row_ranges {\n' +
                '  start_key_closed: "row3"\n' +
                '  end_key_open: "row5"\n' +
                '}\n')
        self.assertEqual(expected, repr(r_set))
