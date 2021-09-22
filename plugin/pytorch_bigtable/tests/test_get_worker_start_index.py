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
from unittest import TestCase
from pbt_C import get_worker_start_index


class IndexGenerationTest(TestCase):
  def test_empty(self):
    for i in range(1, 4):
      for j in range(i):
        self.assertEqual(get_worker_start_index(0, i, j), 0)

  def test_one(self):
    self.assertEqual(get_worker_start_index(1, 1, 0), 0)
    self.assertEqual(get_worker_start_index(1, 1, 1), 1)
    self.assertEqual(get_worker_start_index(1, 2, 0), 0)
    self.assertEqual(get_worker_start_index(1, 2, 1), 1)

  def test_many(self):
    self.assertEqual(get_worker_start_index(10, 1, 0), 0)
    self.assertEqual(get_worker_start_index(10, 1, 1), 10)
    self.assertEqual(get_worker_start_index(10, 2, 0), 0)
    self.assertEqual(get_worker_start_index(10, 2, 1), 5)
    self.assertEqual(get_worker_start_index(10, 2, 2), 10)

  def test_size_of_chunks(self):
    length = 10
    num_workers = 3
    chunks = [(get_worker_start_index(length, num_workers, i),
               get_worker_start_index(length, num_workers, i + 1)) for i in
              range(num_workers)]

    chunk_lengths = [y - x for x, y in chunks]
    self.assertLessEqual(max(chunk_lengths) - min(chunk_lengths), 1)
    self.assertLessEqual(max(chunk_lengths),
                         (length + num_workers - 1) // num_workers)
    self.assertGreaterEqual(min(chunk_lengths), length // num_workers)

  def test_exactly_enough_workers(self):
    length = 5
    num_workers = length
    chunks = [(get_worker_start_index(length, num_workers, i),
               get_worker_start_index(length, num_workers, i + 1)) for i in
              range(num_workers)]

    chunk_lengths = [y - x for x, y in chunks]

    self.assertEqual(chunk_lengths, [1 for _ in range(length)])

  def test_too_many_workers(self):
    length = 5
    num_workers = length * 2
    chunks = [(get_worker_start_index(length, num_workers, i),
               get_worker_start_index(length, num_workers, i + 1)) for i in
              range(num_workers)]

    self.assertEqual(chunks[length:], [(length, length) for _ in range(length)])

    chunk_lengths = [y - x for x, y in chunks]
    self.assertEqual(chunk_lengths[:length], [1 for _ in range(length)])
    self.assertTrue(chunk_lengths[length:], [0 for _ in range(length)])
