# disable module docstring for tests
# pylint: disable=C0114
# disable class docstring for tests
# pylint: disable=C0115
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
    ten[1, 0] = float("NaN")
    ten[5, 0] = float("NaN")
    ten[3, 1] = float("NaN")
    ten[8, 1] = float("NaN")

    client = BigtableClient("fake_project", "fake_instance",
                            endpoint=self.emulator.get_addr())
    table = client.get_table("test-table")

    table.write_tensor(ten, ["fam1:col1", "fam2:col2"],
                       ["row" + str(i).rjust(3, "0") for i in range(20)])
    results = []
    for tensor in table.read_rows(torch.float32, ["fam1:col1", "fam2:col2"],
                                  row_set.from_rows_or_ranges(
                                    row_range.infinite())):
      results.append(tensor.reshape(1, -1))
    result = torch.cat(results)
    self.assertTrue((result.isnan() == ten.isnan()).all().item())
    self.assertTrue((result.nan_to_num(0) == ten.nan_to_num(0)).all().item())

  def test_read_nan(self):
    os.environ["BIGTABLE_EMULATOR_HOST"] = self.emulator.get_addr()
    self.emulator.create_table("fake_project", "fake_instance", "test-table",
                               ["fam1", "fam2"])

    ten = torch.Tensor(list(range(40))).reshape(20, 2)

    client = BigtableClient("fake_project", "fake_instance",
                            endpoint=self.emulator.get_addr())
    table = client.get_table("test-table")

    table.write_tensor(ten, ["fam1:col1", "fam2:col2"],
                       ["row" + str(i).rjust(3, "0") for i in range(20)])
    results = []
    for tensor in table.read_rows(torch.float32, ["fam1:col1", "fam1:col2"],
                                  row_set.from_rows_or_ranges(
                                    row_range.infinite())):
      results.append(tensor.reshape(1, -1))
    result = torch.cat(results)
    self.assertTrue(result[:, 1].isnan().all().item())
    self.assertTrue((result[:, 0] == ten[:, 0]).all().item())

  def test_read_int64(self):
    os.environ["BIGTABLE_EMULATOR_HOST"] = self.emulator.get_addr()
    self.emulator.create_table("fake_project", "fake_instance", "test-table",
                               ["fam1", "fam2"])

    ten = torch.Tensor(list(range(40))).reshape(20, 2).long()

    client = BigtableClient("fake_project", "fake_instance",
                            endpoint=self.emulator.get_addr())
    table = client.get_table("test-table")

    table.write_tensor(ten, ["fam1:col1", "fam2:col2"],
                       ["row" + str(i).rjust(3, "0") for i in range(20)])

    results = []
    for tensor in table.read_rows(torch.int64, ["fam1:col1", "fam1:col2"],
                                  row_set.from_rows_or_ranges(
                                    row_range.infinite()), default_value=0):
      results.append(tensor.reshape(1, -1))
    result = torch.cat(results)
    ten[:, 1] = 0
    different_elements = (ten != result).sum().item()
    self.assertEqual(different_elements, 0)
