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

"""Module containing core functionality of pytorch bigtable dataset"""
import torch
from . import pbt_C
from typing import List, Union, Callable
import pytorch_bigtable.version_filters as filters


class BigtableCredentials:
  pass


class ServiceAccountJson(BigtableCredentials):
  """A class instructing CloudBigtableClient to use a service account."""

  def __init__(self, json_text: str):
    self._json_text = json_text

  @classmethod
  def read_from_file(cls, path: str):
    with open(path, "r", encoding="UTF-8") as f:
      return cls(f.read())


class BigtableClient:
  """CloudBigtableClient is the main entrypoint for
  interacting with the Cloud Bigtable API from PyTorch.

  It encapsulates a connection to Cloud Bigtable and contains
  an accessor for the CloudBigtableReadSession described below.
  """

  def __init__(self, project_id: str, instance_id: str,
               credentials: BigtableCredentials = None,
               endpoint: str = None, ) -> None:
    """Creates a BigtableClient object storing details about the connection.

    Args:
        project_id (str): The assigned project ID of the project.
        instance_id (str): The assigned instance ID.
        credentials (BigtableCredentials): An object used for obtaining
            credentials to authenticate to Cloud Bigtable. If set to None,
            the default credentials will be used (i.e. machine service
            account in GCS or GOOGLE_APPLICATION_CREDENTIALS environment
            variable). Consult google-cloud-cpp project for more
            information.
        endpoint (str): A custom URL, where Cloud Bigtable is available. If
            set to None, the default will be used.
    """
    self._project_id = project_id
    self._instance_id = instance_id
    self._credentials = credentials
    self._endpoint = endpoint

  def get_table(self, table_id: str, app_profile_id: str = None):
    """Creates an instance of BigtableTable

    Args:
        table_id (str): the ID of the table.
        app_profile_id (str): The assigned application profile ID. Defaults
        to None.
    Returns:
        BigtableTable: The relevant table operated through this client.
    """
    return BigtableTable(self, table_id, app_profile_id)


class BigtableTable:
  """Entry point for reading data from Cloud Bigtable.

      Prefetches the sample_row_keys and creates a list with them.
      Each sample is a range open from the right, represented as a pair of two
      row_keys: the first key included in the sample and the first that is
      too big.
  """

  def __init__(self, client: BigtableClient, table_id: str,
               app_profile_id: str = None, ) -> None:
    """
    Args:
        table_id (str): The ID of the table.
        app_profile_id (str): The assigned application profile ID.
        client (BigtableClient): The client on which to operate.
    """
    self._client = client
    self._table_id = table_id
    self._app_profile_id = app_profile_id
    self._sample_row_keys = pbt_C.sample_row_keys(self._client, self._table_id,
                                                  self._app_profile_id)

  def write_tensor(self, tensor: torch.Tensor, columns: List[str],
                   row_keys: Union[
                     List[str], Callable[[torch.Tensor, int], str]]):
    """Opens a connection and writes data from tensor. Each row of this
    tensor will become a row in Bigtable so you should provide as many
    row-keys as tensor.shape(1). Please note that using this function is
    strongly discouraged for large amounts of data. Because it creates a new
    connection every time it is called, it has a non-trivial constant cost,
    so calling it thousands times is not the greatest idea.

    Args:
        tensor: Two dimensional PyTorch Tensor.
        columns: List with names of the columns in the table that
            should be read, i.e:
            [ "column_family_a:column_name_a",
            "column_family_a:column_name_b",
            ...]
        row_keys: a list or a callback.
          If a list, it is a set of row_keys
          that should be used for the rows in the tensor.
          If a callback, it is called with the `tensor`'s row and index and is
          expected to return a row_key for that row.

    """
    if tensor.dim() != 2:
      raise ValueError("`tensor` must have exactly two dimensions")

    if isinstance(row_keys, List) and len(row_keys) != tensor.shape[0]:
      raise ValueError(
        "`row_keys` must have the same length as tensor.shape[0]")

    if len(columns) != tensor.shape[1]:
      raise ValueError("`columns` must have the same length as tensor.shape[1]")

    for i, column_id in enumerate(columns):
      if len(column_id.split(":")) != 2:
        raise ValueError(f"`columns[{i}]` must be a string in format:"
                         " \"column_family:column_name\"")
    row_key_list = None
    row_key_callable = None
    if callable(row_keys):
      row_key_callable = row_keys
    else:
      row_key_list = row_keys

    pbt_C.write_tensor(self._client, self._table_id, self._app_profile_id,
                       tensor, columns, row_key_list, row_key_callable)

  def read_rows(self, cell_type: torch.dtype, columns: List[str],
                row_set: pbt_C.RowSet,
                versions: pbt_C.Filter = filters.latest(), default_value: Union[
        int, float] = None) -> torch.utils.data.IterableDataset:
    """Returns a `CloudBigtableIterableDataset` object.

    Args:
        cell_type (torch.dtype): the type as which to interpret the data in
            the cells
        columns (List[str]): the list of columns to read from; the order on
            this list will determine the order in the output tensors
        row_set (RowSet): set of rows to read.
        versions (Filter):
            specifies which version should be retrieved. Defaults to latest.
        default_value (float|int): value to fill missing values with.
    """

    return _BigtableDataset(self, columns, cell_type, row_set, versions,
                            default_value)


class _BigtableDataset(torch.utils.data.IterableDataset):
  """Dataset that handles iterating over BigTable."""

  def __init__(self, table: BigtableTable, columns: List[str],
               cell_type: torch.dtype, row_set: pbt_C.RowSet,
               versions: pbt_C.Filter = filters.latest(),
               default_value: Union[int, float] = None) -> None:
    super(_BigtableDataset).__init__()

    self._table = table
    self._columns = columns
    self._cell_type = cell_type
    self._row_set = row_set
    self._versions = versions
    self._default_value = default_value

  def __iter__(self):
    """
    Returns an iterator over the CloudBigtable data.

    When called from the main thread we disregard
    the sample_row_keys and perform a single ReadRows call.

    When called from a worker thread each worker calculates
    its share of the row_keys in a deterministic manner and
    downloads them in one API call.
    """

    worker_info = torch.utils.data.get_worker_info()
    num_workers = worker_info.num_workers if worker_info is not None else 1
    worker_id = worker_info.id if worker_info is not None else 0

    return pbt_C.Iterator(self._table._client, self._table._table_id,
                          self._table._app_profile_id,
                          self._table._sample_row_keys, self._columns,
                          self._cell_type, self._row_set, self._versions,
                          self._default_value, num_workers, worker_id)
