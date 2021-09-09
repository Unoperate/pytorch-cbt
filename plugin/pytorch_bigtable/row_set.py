import pbt_C

class RowRange:
    """Represents a continuous range of Bigtable rows.

    These objects are implemented using the C++ counterpart.
    """
    def __init__(self, impl):
        """Not for public use."""
        self._impl = impl

    def infinite():
        """Create a infinite row range."""
        return RowRange(pbt_C.infinite_row_range())

    def starting_at(row_key: str):
        """Create a row range staring at given row (inclusive).

        Args:
            row_key (str): The starting row key of the range (inclusive).
        Returns:
            RowRange: The row range which starts at `row_key` (inclusive).
        """
        return RowRange(pbt_C.starting_at_row_range(row_key))

    def ending_at(row_key: str):
        """Create a row range ending at given row (inclusive).

        Args:
            row_key (str): The ending row key of the range (inclusive).
        Returns:
            RowRange: The row range which ends at `row_key` (inclusive).
        """
        return RowRange(pbt_C.ending_at_row_range(row_key))

    def empty():
        """Create an empty row range."""
        return RowRange(pbt_C.empty_row_range())

    def prefix(prefix: str):
        """Create a range of all rows starting with a given prefix.

        Args:
            prefix (str): The prefix with which all rows start
        Returns:
            RowRange: The row range of all rows starting with the given prefix.
        """
        return RowRange(pbt_C.prefix_row_range(prefix))

    def right_open(start: str, end: str):
        """Create a row range exclusive at the start and inclusive at the end.

        Args:
            start (str): The start of the row range (inclusive).
            end (str): The end of the row range (exclusive).
        Returns:
            RowRange: The row range between the `start` and `end`.
        """
        return RowRange(pbt_C.right_open_row_range(start, end))

    def left_open(start: str, end: str):
        """Create a row range inclusive at the start and exclusive at the end.

        Args:
            start (str): The start of the row range (exclusive).
            end (str): The end of the row range (inclusive).
        Returns:
            RowRange: The row range between the `start` and `end`.
        """
        return RowRange(pbt_C.left_open_row_range(start, end))

    def open(start: str, end: str):
        """Create a row range exclusive at both the start and the end.

        Args:
            start (str): The start of the row range (exclusive).
            end (str): The end of the row range (exclusive).
        Returns:
            RowRange: The row range between the `start` and `end`.
        """
        return RowRange(pbt_C.open_row_range(start, end))

    def closed(start: str, end: str):
        """Create a row range inclusive at both the start and the end.

        Args:
            start (str): The start of the row range (inclusive).
            end (str): The end of the row range (inclusive).
        Returns:
            RowRange: The row range between the `start` and `end`.
        """
        return RowRange(pbt_C.closed_row_range(start, end))

    def __repr__(self):
        return pbt_C.print_row_range(self._impl)

class RowSet:
    """Represents a set of rows described by individual rows and row ranges.

    These objects are implemented using the C++ counterpart.
    """

    def __init__(self, *args):
        """Create a RowSet by summing all arguments - be it rows or row ranges.

        Args:
            *args: A list of individual rows (str) and row ranges (RowRange)
                which the created row set will contain
        """
        self._impl = pbt_C.RowSet()
        for a in args:
            self.append(a)

    def append(self, row_or_range):
        """ Modify this RowSet by appending a row or a row range to it.

        Args:
            row_or_range: A row (str) or row range (RowRange) which will be
                appended to the row set
        """
        if isinstance(row_or_range, RowRange):
            self._impl.append_range(row_or_range._impl)
        else:
            self._impl.append_row(row_or_range)

    def intersect(self, row_range : RowRange):
        """ Modify this RowSet by intersecting its contents with a row range.

        All rows intersecting with the given range will be removed from the set
        and all row ranges will either be adjusted so that they do not cover
        anything beyond the given range or removed entirely (if they have an
        empty intersection with the given range).

        Args:
            row_range (RowRange): The with which this row set will be
                intersected.
        """
        self._impl = self._impl.intersect(row_range._impl)

    def __repr__(self):
        return pbt_C.print_row_set(self._impl)

