import torch
import pbt_C
from typing import List
from pytorch_bigtable.recreate_on_fork import RecreateOnFork

class BigtableCredentials:
    pass

class ServiceAccountJson(BigtableCredentials):
    """A class instructing CloudBigtableClient to use a service account."""

    def __init__(self, json_text : str):
        self._json_text = json_text

    @classmethod
    def read_from_file(cls, path : str):
        with open(path, 'r') as f:
            return cls(f.read())

class BigtableClient:
    """CloudBigtableClient is the main entrypoint for
    interacting with the Cloud Bigtable API from PyTorch.

    It encapsulates a connection to Cloud Bigtable and contains
    an accessor for the CloudBigtableReadSession described below.
    """

    def __init__(self,
            project_id : str,
            instance_id : str,
            credentials : BigtableCredentials=None,
            endpoint : str=None,
            ) -> None:
        """Creates a BigtableClient to start.

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
        self._impl = RecreateOnFork(lambda: pbt_C.create_data_client(project_id,
            instance_id, credentials, endpoint))

    def get_table(self, table_id : str, app_profile_id : str=None):
        """Creates an instance of BigtableTable

        Args:
            table_id (str): the ID of the table.
            app_profile_id (str): The assigned application profile ID. Defaults to None.
        Returns:
            BigtableTable: The relevant table operated through this client.
        """
        return BigtableTable(self, table_id, app_profile_id)

class BigtableTable:
    """Entry point for reading data from Cloud Bigtable.

        Prefetches the sample_row_keys and creates a list with them.
        Each sample is a range open from the right, represented as a pair of two
        row_keys: the first key included in the sample and the first that is too big.
    """

    def __init__(self,
            client : BigtableClient,
            table_id : str,
            app_profile_id : str=None,
            ) -> None:
        """
        Args:
            table_id (str): The ID of the table.
            app_profile_id (str): The assigned application profile ID.
            client (BigtableClient): The client on which to operate.
        """
        self._client = client
        self._table_id = table_id
        self._app_profile_id = app_profile_id
        self._sample_row_keys = pbt_C.sample_row_keys(self._client._impl.get(),
                self._table_id,
                self._app_profile_id)

    def write_tensor(self,
            tensor : torch.Tensor,
            columns : List[str],
            row_keys : List[str],
            ):
        """Opens a connection and writes data from tensor.

        Args:
            tensor: Two dimentional PyTorch Tensor.
            columns: List with names of the columns in the table that
                should be read, i.e:
                [ "column_family_a:column_name_a",
                "column_family_a:column_name_b",
                ...]
            row_keys: list of row_keys that should be used for the rows in the tensor.

        """

        if not len(columns) != tensor.shape[0]:
            raise ValueError(
                "`columns` must have the same length as tensor.shape[0]")

        for i, column_id in enumerate(columns):
            if len(column_id.split(':')) != 2:
                raise ValueError(f"`columns[{i}]` must be a string in format:"
                                 " \"column_family:column_name\"")

        pbt_C.write_tensor(self._client._impl.get(),
                self._table_id,
                self._app_profile_id,
                tensor,
                columns,
                row_keys)

    def read_rows(self,
            cell_type : torch.dtype,
            columns : List[str],
            row_set : pbt_C.RowSet,
            versions : str="latest"
            ) -> torch.utils.data.IterableDataset:
        """Returns a `CloudBigtableIterableDataset` object.

        Args:
            cell_type (torch.dtype): the type as which to interpret the data in
                the cells
            columns (List[str]): the list of columns to read from; the order on
                this list will determine the order in the output tensors
            row_set (RowSet): set of rows to read.
            versions (str):
                specifies which version should be retrieved. Defaults to "latest"
                    "latest": most recent value is returned
                    "oldest": the oldest present value is returned.
        """

        return _BigtableDataset(self, columns, cell_type, row_set, versions)


class _BigtableDataset(torch.utils.data.IterableDataset):

    def __init__(self,
            table : BigtableTable,
            columns : List[str],
            cell_type : torch.dtype,
            row_set : pbt_C.RowSet,
            versions : str) -> None:
        super(_BigtableDataset).__init__()

        self._table = table
        self._columns = columns
        self._cell_type = cell_type
        self._row_set = row_set
        self._versions = versions

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
        worker_id = worker_info.worker_id if worker_info is not None else 0

        return pbt_C.Iterator(self._table._client._impl.get(),
                self._table._table_id,
                self._table._app_profile_id,
                self._table._sample_row_keys,
                self._columns,
                self._cell_type,
                self._row_set,
                self._versions,
                num_workers,
                worker_id)
