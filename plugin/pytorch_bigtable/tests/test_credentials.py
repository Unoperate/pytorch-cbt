# disable module docstring for tests
# pylint: disable=C0114
# disable class docstring for tests
# pylint: disable=C0115
# disable warning for access to protected members
# pylint: disable=W0212
from unittest import TestCase
from tempfile import NamedTemporaryFile
from pytorch_bigtable.bigtable_dataset import ServiceAccountJson


class ServiceAccountJsonTest(TestCase):
  def test_reading_from_file(self):
    with NamedTemporaryFile(buffering=0) as tmpfile:
      tmpfile.write("example_content".encode())
      json_creds = ServiceAccountJson.read_from_file(tmpfile.name)
      self.assertEqual("example_content", json_creds._json_text)
