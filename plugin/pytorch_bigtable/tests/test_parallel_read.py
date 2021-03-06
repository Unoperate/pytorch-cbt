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
# disable warning for accessing protected members
# pylint: disable=W0212
import unittest
import torch
import os
from .bigtable_emulator import BigtableEmulator
from pytorch_bigtable import BigtableClient, row_set, row_range
from torch.utils.data import DataLoader


class BigtableParallelReadTest(unittest.TestCase):
  def setUp(self):
    self.emulator = BigtableEmulator()

  def tearDown(self):
    self.emulator.stop()

  def test_read(self):
    os.environ['BIGTABLE_EMULATOR_HOST'] = self.emulator.get_addr()

    num_rows = 300

    self.emulator.create_table('fake_project', 'fake_instance', 'test-table',
                               ['fam1', 'fam2'],
                               ['row' + str(i).rjust(3, '0') for i in
                                range(0, num_rows, 30)])

    ten = torch.Tensor(list(range(num_rows * 2))).reshape(num_rows, 2)

    client = BigtableClient('fake_project', 'fake_instance',
                            endpoint=self.emulator.get_addr())
    table = client.get_table('test-table')

    table.write_tensor(ten, ['fam1:col1', 'fam2:col2'],
                       ['row' + str(i).rjust(3, '0') for i in range(num_rows)])

    table = client.get_table('test-table')

    ds = table.read_rows(torch.float32, ['fam1:col1', 'fam2:col2'],
                         row_set.from_rows_or_ranges(row_range.infinite()))

    loader = DataLoader(ds, num_workers=4)
    output = []
    for tensor in loader:
      output.append(tensor)
    output = sorted(output, key=lambda x: x[0, 0].item())
    output = torch.cat(output)
    self.assertTrue((ten == output).all())

  def test_sample_row_keys(self):
    os.environ['BIGTABLE_EMULATOR_HOST'] = self.emulator.get_addr()
    self.emulator.create_table('fake_project', 'fake_instance', 'test-table',
                               ['fam1', 'fam2'],
                               ['row' + str(i).rjust(3, '0') for i in
                                range(0, 500, 50)])

    ten = torch.Tensor(list(range(500))).reshape(250, 2)

    client = BigtableClient('fake_project', 'fake_instance',
                            endpoint=self.emulator.get_addr())
    table = client.get_table('test-table')

    table.write_tensor(ten, ['fam1:col1', 'fam2:col2'],
                       ['row' + str(i).rjust(3, '0') for i in range(250)])

    table = client.get_table('test-table')
    self.assertGreater(len(table._sample_row_keys), 0)
