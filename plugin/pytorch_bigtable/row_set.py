import pbt_C
from typing import Union


def empty() -> pbt_C.RowSet:
  """Create an empty row set."""
  return pbt_C.RowSet()


def append(row_set: pbt_C.RowSet,
           row_or_range: Union[str, pbt_C.RowRange]) -> pbt_C.RowSet:
  """ Modify a row_set by appending a row or a row range to it.

  Args:
    row_set: A set (RowSet) to append to.
    row_or_range: A row (str) or row range (RowRange) which will be
        appended to the row set.
  Returns:
    RowSet: The original row set expanded by the given `row_or_range`.
  """
  if isinstance(row_or_range, pbt_C.RowRange):
    return row_set.append_range(row_or_range)
  else:
    return row_set.append_row(row_or_range)


def from_row_range(row_range: pbt_C.RowRange) -> pbt_C.RowSet:
  """ Create a set from a row range.

  Args:
    row_range: A row range (RowRange) which will be
        appended to an empty row set.
  Returns:
    RowSet: a set of rows containing the given row range.
  """
  return append(empty(), row_range)


def intersect(row_set, row_range: pbt_C.RowRange) -> pbt_C.RowSet:
  """ Modify a row set by intersecting its contents with a row range.

  All rows intersecting with the given range will be removed from the set
  and all row ranges will either be adjusted so that they do not cover
  anything beyond the given range or removed entirely (if they have an
  empty intersection with the given range).

  Args:
    row_set: A set (RowSet) which will be intersected.
    row_range (RowRange): The range with which this row set will be
        intersected.
  Returns:
    RowSet: an intersection of the given row set and row range.
  """
  return row_set.intersect(row_range)
