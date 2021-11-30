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
# disable unused parameter for callback
# pylint: disable=W0613
import random
import datetime
import unittest
import torch
import os
from .bigtable_emulator import BigtableEmulator
from pytorch_bigtable import BigtableClient, row_set, row_range
from google.auth.credentials import AnonymousCredentials
from google.cloud.bigtable import Client


def check_values(test_case, values, table, type_name, dtype):
    for i, r in enumerate(
        table.read_rows(
            dtype,
            ["fam1:" + type_name],
            row_set=row_set.from_rows_or_ranges(row_range.infinite()),
        )
    ):
        test_case.assertEqual(values[i].item(), r[0].item())


def write_and_check_values(test_case, values, table, type_name, dtype):

    table.write_tensor(
        values.reshape(-1,1), ["fam1:" + type_name + "_serialized"], test_case.data['row_keys'])

    check_values(test_case, values, table, type_name + "_serialized", dtype)


class BigtableSerializationTest(unittest.TestCase):
    def setUp(self):
        self.emulator = BigtableEmulator()
        self.data = {
            "values": [i * 10 / 7 for i in range(10)],
            "row_keys": ["row" + str(i).rjust(3, "0") for i in range(10)],
            "float": [
                b"\x00\x00\x00\x00",
                b"?\xb6\xdbn",
                b"@6\xdbn",
                b"@\x89$\x92",
                b"@\xb6\xdbn",
                b"@\xe4\x92I",
                b"A\t$\x92",
                b"A \x00\x00",
                b"A6\xdbn",
                b"AM\xb6\xdb",
            ],
            "double": [
                b"\x00\x00\x00\x00\x00\x00\x00\x00",
                b"?\xf6\xdbm\xb6\xdbm\xb7",
                b"@\x06\xdbm\xb6\xdbm\xb7",
                b"@\x11$\x92I$\x92I",
                b"@\x16\xdbm\xb6\xdbm\xb7",
                b"@\x1c\x92I$\x92I%",
                b"@!$\x92I$\x92I",
                b"@$\x00\x00\x00\x00\x00\x00",
                b"@&\xdbm\xb6\xdbm\xb7",
                b"@)\xb6\xdbm\xb6\xdbn",
            ],
            "int32": [
                b"\x00\x00\x00\x00",
                b"\x00\x00\x00\x01",
                b"\x00\x00\x00\x02",
                b"\x00\x00\x00\x04",
                b"\x00\x00\x00\x05",
                b"\x00\x00\x00\x07",
                b"\x00\x00\x00\x08",
                b"\x00\x00\x00\n",
                b"\x00\x00\x00\x0b",
                b"\x00\x00\x00\x0c",
            ],
            "int64": [
                b"\x00\x00\x00\x00\x00\x00\x00\x00",
                b"\x00\x00\x00\x00\x00\x00\x00\x01",
                b"\x00\x00\x00\x00\x00\x00\x00\x02",
                b"\x00\x00\x00\x00\x00\x00\x00\x04",
                b"\x00\x00\x00\x00\x00\x00\x00\x05",
                b"\x00\x00\x00\x00\x00\x00\x00\x07",
                b"\x00\x00\x00\x00\x00\x00\x00\x08",
                b"\x00\x00\x00\x00\x00\x00\x00\n",
                b"\x00\x00\x00\x00\x00\x00\x00\x0b",
                b"\x00\x00\x00\x00\x00\x00\x00\x0c",
            ],
            "bool": [
                b'\x00',
                b'\xff',
                b'\xff',
                b'\xff',
                b'\xff',
                b'\xff',
                b'\xff',
                b'\xff',
                b'\xff',
                b'\xff'
            ],
        }

        os.environ["BIGTABLE_EMULATOR_HOST"] = self.emulator.get_addr()
        self.emulator.create_table(
            "fake_project", "fake_instance", "test-table", ["fam1"]
        )

        client = Client(
            project="fake_project", credentials=AnonymousCredentials(), admin=True
        )
        table = client.instance("fake_instance").table("test-table")

        for type_name in ["float", "double", "int32", "int64", "bool"]:
            rows = []
            for i, value in enumerate(self.data[type_name]):
                row_key = self.data['row_keys'][i]
                row = table.direct_row(row_key)
                row.set_cell(
                    "fam1", type_name, value, timestamp=datetime.datetime.utcnow()
                )
                rows.append(row)
            table.mutate_rows(rows)

    def tearDown(self):
        self.emulator.stop()

    def test_read_float(self):
        values = torch.DoubleTensor(self.data['values']).type(torch.float32)

        client = BigtableClient("fake_project", "fake_instance")
        table = client.get_table("test-table")

        check_values(self, values, table, "float", torch.float32)

    def test_read_double(self):
        values = torch.DoubleTensor(self.data["values"]).type(torch.float64)

        client = BigtableClient("fake_project", "fake_instance")
        table = client.get_table("test-table")

        check_values(self, values, table, "double", torch.float64)

    def test_read_int64(self):
        values = torch.DoubleTensor(self.data["values"]).type(torch.int64)

        client = BigtableClient("fake_project", "fake_instance")
        table = client.get_table("test-table")

        check_values(self, values, table, "int64", torch.int64)

    def test_read_int32(self):
        values = torch.DoubleTensor(self.data["values"]).type(torch.int32)

        client = BigtableClient("fake_project", "fake_instance")
        table = client.get_table("test-table")

        check_values(self, values, table, "int32", torch.int32)

    def test_read_bool(self):
        values = torch.DoubleTensor(self.data["values"]).type(torch.bool)

        client = BigtableClient("fake_project", "fake_instance")
        table = client.get_table("test-table")

        check_values(self, values, table, "bool", torch.bool)

    def test_write_float(self):
        column_name = "serialized_float"
        values = torch.DoubleTensor(self.data['values']).type(torch.float32)

        client = BigtableClient("fake_project", "fake_instance")
        table = client.get_table("test-table")

        table.write_tensor(
            values, ["fam1:" + column_name], self.data['row_keys'])

        write_and_check_values(self, values, table, column_name, torch.float32)

    def test_write_float(self):
        values = torch.DoubleTensor(self.data['values']).type(torch.float32)

        client = BigtableClient("fake_project", "fake_instance")
        table = client.get_table("test-table")

        write_and_check_values(self, values, table, "float", torch.float32)

    def test_write_double(self):
        values = torch.DoubleTensor(self.data["values"]).type(torch.float64)

        client = BigtableClient("fake_project", "fake_instance")
        table = client.get_table("test-table")

        write_and_check_values(self, values, table, "double", torch.float64)

    def test_write_int64(self):
        values = torch.DoubleTensor(self.data["values"]).type(torch.int64)

        client = BigtableClient("fake_project", "fake_instance")
        table = client.get_table("test-table")

        write_and_check_values(self, values, table, "int64", torch.int64)

    def test_write_int32(self):
        values = torch.DoubleTensor(self.data["values"]).type(torch.int32)

        client = BigtableClient("fake_project", "fake_instance")
        table = client.get_table("test-table")

        write_and_check_values(self, values, table, "int32", torch.int32)

    def test_write_bool(self):
        values = torch.DoubleTensor(self.data["values"]).type(torch.bool)

        client = BigtableClient("fake_project", "fake_instance")
        table = client.get_table("test-table")

        write_and_check_values(self, values, table, "bool", torch.bool)
