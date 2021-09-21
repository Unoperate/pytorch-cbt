# pylint: disable=C0114
# pylint: disable=C0115
# pylint: disable=W0212
# pylint: disable=R1732
import unittest
import torch
import os
from .bigtable_emulator import BigtableEmulator
from pytorch_bigtable import BigtableClient, row_set, row_range


class BigtableReadTest(unittest.TestCase):
  def setUp(self):
    self.emulator = BigtableEmulator()

  def tearDown(self):
    self.emulator.stop()

  def test_read(self):
    os.environ["BIGTABLE_EMULATOR_HOST"] = self.emulator.get_addr()
    self.emulator.create_table("fake_project", "fake_instance", "test-table",
                               ["fam1", "fam2"])

    ten = torch.Tensor(list(range(40))).reshape(20, 2)

    client = BigtableClient("fake_project", "fake_instance",
                            endpoint=self.emulator.get_addr())
    table = client.get_table("test-table")

    table.write_tensor(ten, ["fam1:col1", "fam2:col2"],
                       ["row" + str(i).rjust(3, "0") for i in range(20)])
    for tensor in table.read_rows(torch.float32, ["fam1:col1", "fam1:col2"],
        row_set.from_rows_or_ranges(row_range.infinite())):
      print(f"Got tensor: {tensor}")
