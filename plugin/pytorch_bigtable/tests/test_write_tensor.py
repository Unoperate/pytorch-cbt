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

# disable module docstring for tests
# pylint: disable=C0114
# disable class docstring for tests
# pylint: disable=C0115
import unittest
import torch
import os
from .bigtable_emulator import BigtableEmulator
from pytorch_bigtable import BigtableClient, row_set, row_range


class BigtableWriteTest(unittest.TestCase):
  def setUp(self):
    self.emulator = BigtableEmulator()

  def tearDown(self):
    self.emulator.stop()

  def test_write_arguments(self):
    os.environ["BIGTABLE_EMULATOR_HOST"] = self.emulator.get_addr()
    self.emulator.create_table("fake_project", "fake_instance", "test-table",
                               ["fam1", "fam2"])

    ten = torch.Tensor(list(range(40))).reshape(20, 2)

    client = BigtableClient("fake_project", "fake_instance",
                            endpoint=self.emulator.get_addr())
    table = client.get_table("test-table")

    # empty columns
    self.assertRaises(ValueError, table.write_tensor, ten, [],
                      ["row" + str(i).rjust(3, "0") for i in range(20)])
    # not enough columns
    self.assertRaises(ValueError, table.write_tensor, ten, ["fam1:c1"],
                      ["row" + str(i).rjust(3, "0") for i in range(20)])
    # too many columns
    self.assertRaises(ValueError, table.write_tensor, ten,
                      ["fam1:c1", "fam1:c2", "fam1:c2"],
                      ["row" + str(i).rjust(3, "0") for i in range(20)])
    # columns without families
    self.assertRaises(ValueError, table.write_tensor, ten, ["c1", "c2"],
                      ["row" + str(i).rjust(3, "0") for i in range(20)])
    # not enough row_keys
    self.assertRaises(ValueError, table.write_tensor, ten,
                      ["fam1:c1", "fam1:c2"],
                      ["row" + str(i).rjust(3, "0") for i in range(10)])

    self.assertRaises(ValueError, table.write_tensor, ten[0],
                      ["fam1:c1", "fam1:c2"], ["row000"])

    # non existing family
    self.assertRaises(RuntimeError, table.write_tensor, ten,
                      ["fam3:c1", "fam3:c2"],
                      ["row" + str(i).rjust(3, "0") for i in range(20)])

  def test_write_single_row(self):
    os.environ["BIGTABLE_EMULATOR_HOST"] = self.emulator.get_addr()
    self.emulator.create_table("fake_project", "fake_instance", "test-table",
                               ["fam1", "fam2"])

    ten = torch.tensor([1, 2], dtype=torch.float32).reshape(1, -1)

    client = BigtableClient("fake_project", "fake_instance",
                            endpoint=self.emulator.get_addr())
    table = client.get_table("test-table")

    table.write_tensor(ten, ["fam1:col1", "fam1:col2"], ["row000"])

    result = next(iter(
      table.read_rows(torch.float32, ["fam1:col1", "fam1:col2"],
                      row_set.from_rows_or_ranges(row_range.infinite()))))

    self.assertTrue((ten == result).all().item())

  def test_write_single_column(self):
    os.environ["BIGTABLE_EMULATOR_HOST"] = self.emulator.get_addr()
    self.emulator.create_table("fake_project", "fake_instance", "test-table",
                               ["fam1", "fam2"])

    ten = torch.Tensor(list(range(40))).reshape(-1, 1)

    client = BigtableClient("fake_project", "fake_instance",
                            endpoint=self.emulator.get_addr())
    table = client.get_table("test-table")

    table.write_tensor(ten, ["fam1:col1"],
                       ["row" + str(i).rjust(3, "0") for i in range(40)])

    results = []
    for tensor in table.read_rows(torch.float32, ["fam1:col1"],
                                  row_set.from_rows_or_ranges(
                                    row_range.infinite()), default_value=0):
      results.append(tensor.reshape(1, -1))
    result = torch.cat(results)
    self.assertTrue((ten == result).all().item())
