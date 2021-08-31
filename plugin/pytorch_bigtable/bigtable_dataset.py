from _typeshed import Self
import torch
import math
import pbt_C


class BigtableClient:
    """CloudBigtableClient is the main entrypoint for
    interacting with the Cloud Bigtable API from PyTorch.

    It encapsulates a connection to Cloud Bigtable and contains
    an accessor for the CloudBigtableReadSession described below.
    """

    def __init__(self,
                 credentials_path=None,
                 endpoint=None
                 ) -> None:
        """Creates a BigtableClient to start.

        Args:
            credentails_path: path to credentials json. If not
            specified, uses the defualt credentials from the environment
            variable.
            endpoint: if not specified, uses the default bigtable endpoint
        """

        # credentials_path
        if credentials_path and not isinstance(credentials_path, str):
            raise ValueError("`credentials_path` must be a string")

        # endpoint
        if endpoint and not isinstance(endpoint, str):
            raise ValueError("`endpoint` must be a string")

        self.credentials_path = credentials_path
        self.endpoint = endpoint


class BigtableTable:
    """Entry point for reading data from Cloud Bigtable.

        Prefetches the samples and creates a list with them.
        Each sample is a range open from the right, represented as a pair of two
        row_keys: the first key included in the sample and the first that is too big.
    """

    def __init__(self,
                 project_id,
                 instance_id,
                 table_id,
                 application_profile_id,
                 client
                 ) -> None:
        """
        Args:
            project_id: The assigned project ID of the project.
            instance_id: The assigned instance ID.
            table_id: The ID of the table.
            application_profile_id: The assigned application profile ID.
            client: BigtableClient object
        """

        # project_id
        if not isinstance(project_id, str):
            raise ValueError("`project_id` must be a string")

        # instance_id
        if not isinstance(instance_id, str):
            raise ValueError("`instance_id` must be a string")

        # table_id
        if not isinstance(table_id, str):
            raise ValueError("`table_id` must be a string")

        # application_profile_id
        if not isinstance(application_profile_id, str):
            raise ValueError("`application_profile_id` must be a string")

        # client
        if not isinstance(client, BigtableClient):
            raise ValueError("`client` must be a BigtableClient")

        self._client = client
        self._project_id = project_id
        self._instance_id = instance_id
        self._table_id = table_id
        self._application_profile_id = application_profile_id
        self._samples = pbt_C.io_big_table_sample_row_key(self._project_id,
                                                          self._instance_id,
                                                          self._table_id,
                                                          self._application_profile_id)

    def write_tensor(self,
                     tensor,
                     columns,
                     row_key_prefix=None,
                     offset=0,
                     row_keys=None,
                     ):
        """Opens a connection and writes data from tensor.

        Args:
            tensor: Two dimentional PyTorch Tensor.
            project_id: The assigned project ID of the project.
            instance_id: The assigned instance ID.
            table_id: The ID of the table.
            application_profile_id: The assigned application profile ID.
            columns: List with names of the columns in the table that
                should be read, i.e:
                [ "column_family_a:column_name_a",
                "column_family_a:column_name_b",
                ...]
            row_key_prefix: If specified the row keys will be generated dynamically
                in format `<row_key_prefix><offset+i>` where i is the row's index.
            offset: If specified the indexes in the row_keys will start at that index.
            row_keys: list of row_keys that should be used for the rows in the tensor.
                This is only needed when row_key_prefix is not specified.

        """

        # tensor
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("`tensor` must be a torch.Tensor")

        # columns
        if not isinstance(columns, list):
            raise ValueError("`columns` must be a list of strings")
        if not len(columns) != tensor.shape[0]:
            raise ValueError(
                "`columns` must have the same length as tensor.shape[0]")

        for i, column_id in enumerate(columns):
            if not isinstance(column_id, str) or len(column_id.split(':')) != 2:
                raise ValueError(f"`columns[{i}]` must be a string in format:"
                                 " \"column_family:column_name\"")

        if not row_keys:
            # row_key_prefix and offset
            if not row_key_prefix:
                raise ValueError(
                    "either `row_keys` or `row_key_prefix` must be set")

            if not isinstance(row_key_prefix, str):
                raise ValueError("`row_key_prefix` must be a string")

            if not isinstance(offset, int) or offset < 0:
                raise ValueError("`offset` must be a nonnegative integer")
        else:
            # row_keys
            if not isinstance(row_keys, list):
                raise ValueError(f"`row_keys` must be a list of strings")

            if len(row_keys) != tensor.shape[1]:
                raise ValueError(f"`row_keys` must have the same length"
                                 " as tensor.shape[1]")

            for i, row_key in enumerate(row_keys):
                if not isinstance(row_key, str):
                    raise ValueError(f"`row_keys[{i}]` must be a string")

        pbt_C.io_big_table_write(self._project_id,
                                 self._instance_id,
                                 self._table_id,
                                 self._application_profile_id,
                                 tensor,
                                 columns,
                                 row_key_prefix,
                                 offset,
                                 row_keys)

    def read_rows_dataset(self,
                          selected_columns,
                          output_types=None,
                          start_key=None,
                          end_key=None,
                          row_key_prefix=None,
                          versions=1
                          ):
        """Returns a `CloudBigtableIterableDataset` object.

        Args:
            selected_columns: This can be a list or a dict. If a list, it has
                names of the columns in the table that should be read. If a dict,
                it should be in a form like, i.e:
                { "column_family_a:column_name_a": { output_type: torch.int64},
                "column_family_a:column_name_b": { output_type: torch.float32},
                ...
                "column_family_a:column_name_x": { output_type: torch.float32},
                "column_family_b:column_name_a": { output_type: torch.float32},
                ...
                }
                If "output_type" not specified, INT64 is implied for all Tensors.
            output_types: Types for the output tensor in the same sequence as
                selected_columns. This is only needed when selected_columns is a list,
                if selected_columns is a dictionary, this output_types information is
                included in selected_columns as described above.
                If not specified, INT64 is implied for all Tensors.
            start_key: row key range. Reading data from rows from <start key> to
                <end key>. The range is assumed to be half-open, where the start
                key is included and the end key is the first excluded key after
                the range. Both keys are optional - if both are left out then
                a full table scan will be performed.
            end_key: the first excluded key after the range.
            row_key_prefix: read data from all rows with a given prefix.
            versions: specific number of versions to be returned. By default only
                the latest version of a column shall be returned. If `versions` is
                specified, values' versions will be returned as the third dimension.
                If a value has less versions than specified, missing values are
                replaced with NaNs.
        """

        # selected_columns
        if not isinstance(selected_columns, list):
            raise ValueError("`selected_columns` must be a list of strings")

        for i, column_id in enumerate(selected_columns):
            if not isinstance(column_id, str) or len(column_id.split(':')) != 2:
                raise ValueError(f"`selected_columns[{i}]` must be a string in format:"
                                 " \"column_family:column_name\"")

        # output types
        if output_types:
            if not isinstance(output_types, list):
                raise ValueError(
                    "`output_types` must be a list if selected_columns is list"
                )
            if len(output_types) != len(selected_columns):
                raise ValueError(
                    "lengths of `output_types` must be a same as the "
                    "length of `selected_columns`"
                )
        else:
            output_types = [torch.int64] * len(selected_columns)

        # start_key
        if start_key and not isinstance(start_key, str):
            raise ValueError("`start_key` must be a string")

        # end_key
        if end_key and not isinstance(end_key, str):
            raise ValueError("`end_key` must be a string")

        # row_key_prefix
        if row_key_prefix and not isinstance(row_key_prefix, str):
            raise ValueError("`row_key_prefix` must be a string")

        # versions
        if not isinstance(versions, int) or versions < 1:
            raise ValueError("`versions` must be a positive integer")

        return _BigtableDataset(self._client,
                                self._project_id,
                                self._instance_id,
                                self._table_id,
                                self._application_profile_id,
                                self._samples,
                                selected_columns,
                                start_key,
                                end_key,
                                row_key_prefix,
                                versions)


class _BigtableDataset(torch.utils.data.IterableDataset):

    def __init__(self,
                 project_id,
                 instance_id,
                 table_id,
                 application_profile_id,
                 client,
                 samples,
                 selected_columns,
                 start_key,
                 end_key,
                 row_key_prefix,
                 versions) -> None:
        super(_BigtableDataset).__init__()

        self._client = client
        self._project_id = project_id
        self._instance_id = instance_id
        self._table_id = table_id
        self._application_profile_id = application_profile_id
        self._samples = samples
        self._selected_columns = selected_columns
        self._start_key = start_key
        self._end_key = end_key
        self._row_key_prefix = row_key_prefix
        self._versions = versions

    def __iter__(self):
        """
        Returns an iterator over the CloudBigtable data.

        When called from the main thread we disregard
        the samples and perform a single ReadRows call. 

        When called from a worker thread each worker calculates
        its share of the row_keys in a deterministic manner and 
        downloads them in one API call.
        """

        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.worker_id if worker_info is not None else 0

        return pbt_C.io_big_table_iterator(self._project_id,
                                           self._instance_id,
                                           self._table_id,
                                           self._application_profile_id,
                                           self._samples,
                                           self._selected_columns,
                                           self._start_key,
                                           self._end_key,
                                           self._row_key_prefix,
                                           self._versions,
                                           self._client,
                                           num_workers,
                                           worker_id)
