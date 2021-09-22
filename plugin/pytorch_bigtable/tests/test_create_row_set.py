from unittest import TestCase
from pytorch_bigtable import row_set, row_range
from pbt_C import create_row_set


class CreateRowSetTest(TestCase):
  def test_empty(self):
    row_sets = [row_set.from_rows_or_ranges(row_range.infinite()),
                row_set.from_rows_or_ranges(
                  row_range.right_open("row-a", "row-c")),
                row_set.from_rows_or_ranges(
                  row_range.right_open("row-a", "row-c"), "row-g",
                  row_range.right_open("row-h", "row-j"))]
    for rs in row_sets:
      output_0 = create_row_set(rs, [], 2, 0)
      self.assertEqual(repr(output_0), repr(rs))

      output_1 = create_row_set(rs, [], 2, 1)
      self.assertEqual(repr(output_1),
                       repr(row_set.from_rows_or_ranges(row_range.empty())))

  def test_infinite(self):
    row_sets = [
      row_set.from_rows_or_ranges(row_range.right_open("row-a", "row-c")),
      row_set.from_rows_or_ranges(row_range.right_open("row-a", "row-c"),
                                  "row-g",
                                  row_range.right_open("row-h", "row-j"))]
    samples = [('', 10)]
    for rs in row_sets:
      output_0 = create_row_set(rs, samples, 2, 0)
      self.assertEqual(repr(output_0), repr(rs))

      output_1 = create_row_set(rs, samples, 2, 1)
      self.assertTrue(output_1.is_empty())

  def test_intersection(self):
    rs = row_set.from_rows_or_ranges(row_range.right_open("row-a", "row-e"))
    samples = [('row-a', 10), ('row-c', 10), ('row-e', 10), ('row-g', 10)]

    output_0 = create_row_set(rs, samples, 2, 0)
    expected_0 = rs.intersect(row_range.right_open("row-a", "row-c"))
    self.assertEqual(repr(output_0), repr(expected_0))

    output_1 = create_row_set(rs, samples, 2, 1)
    expected_1 = rs.intersect(row_range.right_open("row-c", "row-e"))
    self.assertEqual(repr(output_1), repr(expected_1))

  def test_intersection_empty(self):
    rs = row_set.from_rows_or_ranges(row_range.empty())
    samples = [('row-a', 10), ('row-c', 10), ('row-e', 10), ('row-g', 10)]

    output_0 = create_row_set(rs, samples, 2, 0)
    self.assertTrue(output_0.is_empty())

    output_1 = create_row_set(rs, samples, 2, 1)
    self.assertTrue(output_1.is_empty())

  def test_single_sample(self):
    rs = row_set.from_rows_or_ranges(row_range.infinite())
    samples = [('row-a', 10)]

    output_0 = create_row_set(rs, samples, 2, 0)
    self.assertEqual(repr(output_0), repr(
      row_set.from_rows_or_ranges(row_range.right_open('', 'row-a'))))

    output_1 = create_row_set(rs, samples, 2, 1)
    self.assertEqual(repr(output_1), repr(
      row_set.from_rows_or_ranges(row_range.right_open('row-a', ''))))

  def test_too_many_workers_per_sample(self):
    rs = row_set.from_rows_or_ranges(row_range.infinite())
    samples = [('row-a', 10)]

    output_0 = create_row_set(rs, samples, 3, 0)
    self.assertEqual(repr(output_0), repr(
      row_set.from_rows_or_ranges(row_range.right_open('', 'row-a'))))

    output_1 = create_row_set(rs, samples, 3, 1)
    self.assertEqual(repr(output_1), repr(
      row_set.from_rows_or_ranges(row_range.right_open('row-a', ''))))

    output_2 = create_row_set(rs, samples, 3, 2)
    self.assertTrue(output_2.is_empty())
