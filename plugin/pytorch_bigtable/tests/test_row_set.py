from unittest import TestCase

from pytorch_bigtable.row_set import RowRange,RowSet

class RowRangeTest(TestCase):
    def test_infinite(self):
        self.assertEqual("", repr(RowRange.infinite()))

    def test_starting_at(self):
        expected = 'start_key_closed: "row1"\n'
        self.assertEqual(expected, repr(RowRange.starting_at("row1")))

    def test_ending_at(self):
        expected = 'end_key_closed: "row1"\n'
        self.assertEqual(expected, repr(RowRange.ending_at("row1")))

    def test_empty(self):
        expected = (
                'start_key_open: ""\n' +
                'end_key_open: "\\000"\n')

        self.assertEqual(expected, repr(RowRange.empty()))

    def test_prefix(self):
        expected = (
                'start_key_closed: "row1"\n'
                'end_key_open: "row2"\n'
                )
        self.assertEqual(expected, repr(RowRange.prefix("row1")))

    def test_right_open(self):
        expected = (
                'start_key_closed: "row1"\n'
                'end_key_open: "row2"\n'
                )
        self.assertEqual(expected, repr(RowRange.right_open("row1", "row2")))

    def test_left_open(self):
        expected = (
                'start_key_open: "row1"\n'
                'end_key_closed: "row2"\n'
                )
        self.assertEqual(expected, repr(RowRange.left_open("row1", "row2")))

    def test_open(self):
        expected = (
                'start_key_open: "row1"\n'
                'end_key_open: "row2"\n'
                )
        self.assertEqual(expected, repr(RowRange.open("row1", "row2")))

    def test_closed(self):
        expected = (
                'start_key_closed: "row1"\n' +
                'end_key_closed: "row2"\n')
        self.assertEqual(expected, repr(RowRange.closed("row1", "row2")))

class TestRowSet(TestCase):
    def test_empty(self):
        expected = ''
        self.assertEqual(expected, repr(RowSet()))

    def test_append_row(self):
        row_set = RowSet()
        row_set.append("row1")
        expected = 'row_keys: "row1"\n'
        self.assertEqual(expected, repr(row_set))

    def test_append_row_range(self):
        row_set = RowSet()
        row_set.append(RowRange.closed("row1", "row2"))
        expected = (
                'row_ranges {\n' +
                '  start_key_closed: "row1"\n' +
                '  end_key_closed: "row2"\n' +
                '}\n')
        self.assertEqual(expected, repr(row_set))

    def test_constructor(self):
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

        row_set = RowSet(
                RowRange.closed("row1", "row2"),
                "row3",
                RowRange.open("row4", "row5"),
                "row6")
        self.assertEqual(expected, repr(row_set))

    def test_intersect(self):
        row_set = RowSet(RowRange.open("row1", "row5"))
        row_set.intersect(RowRange.closed("row3", "row7"))
        expected = (
                'row_ranges {\n' +
                '  start_key_closed: "row3"\n' +
                '  end_key_open: "row5"\n' +
                '}\n')
        self.assertEqual(expected, repr(row_set))
