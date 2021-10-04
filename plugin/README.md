# Pytorch Bigtable Extension

This is a Pytorch Extension used to connect to Google Cloud Bigtable.

### Outline:

* Installation
* Quickstart
* Example
* Reading specific rowkeys
* Reading specific versions
* Writing data to BT (callback)
* Specific columns
* parallel read

## Installation

Make sure you have torch installed. Then just use pip to install the latest version

```
pip install -i https://test.pypi.org/simple/ pytorch-bigtable
```
## Credentials
Right now only the default credentials are supported. To connect to Bigtable you need
to set the environment variable `GOOGLE_APPLICATION_CREDENTIALS`. 
Replace `[PATH]` with the file path of the JSON file that contains your service account key.
```python
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="[PATH]"
```
## Quickstart

First you need to create a client and a table you would like to read from.

```python
import torch
import pytorch_bigtable as pbt
import random

client = pbt.BigtableClient(project_id="project-id", instance_id="instance-id")
train_table = client.get_table("train")
```

First we will write some data into Bigtable. To do that, we create a tensor `ten`. We provide a list of column names in
format `column_family:column_name` and a list of rowkeys.

```python
import random

ten = torch.Tensor(list(range(40))).reshape(20, 2)
train_table.write_tensor(ten, ["fam1:col1", "fam2:col2"],
                         ["row" + str(random.randint(0, 999)).rjust(3, "0") for _ in range(20)])
```

Great! Now we can create a pytorch dataset that will read the data from our table. To do that, you have to provide the
type of the data you wish to read, list of column names in format `column_family:column_name`, and a row_set that you
would like to read.

```python
import pytorch_bigtable.row_set
import pytorch_bigtable.row_range

row_set = pbt.row_set.from_rows_or_ranges(pbt.row_range.infinite())

for tensor in train_table.read_rows(torch.float32, ["fam1:col1", "fam2:col2"], row_set):
  print(tensor)
```

That's it! Congrats!
You can also explore our example of training a fraud-detection model on data from Bigtable in example.py

## Parallel read

Our dataset supports reading in parallel from Bigtable. To do that, create a pytorch DataLoader and set num_workers to a
number higher than one. First a list of tablets will be fetched from bigquery, dividing the work into chunks. Then each
worker will compute it's share of work and start reading from their tablets.

**Note**: Keep in mind that when reading in parallel, the rows are not guaranteed to be read in order.

## Reading specific row_keys

To read the data from Bigtable, you can specify a set of rows or a range or a combination of those. We partly expose the
C++ Bigtable Client api for that purpose.

pytorch_bigtable.BigtableTable.read_rows method expects you to provide a row_set. You can construct a row_set from
row_keys or row_ranges as follows:

```python
import pytorch_bigtable.row_set as row_set
import pytorch_bigtable.row_range as row_range

row_range_ah = row_range.right_open("row-a", "row-h")

my_row_set = row_set.from_rows_or_ranges(row_range_ah, "row-x", "row-y")
```

such row_set would contain a range of rows `[a, h)` and rows "row-x" and "row-y".

you can also create a row_set from an infinite range, empty range or a prefix. You can also intersect it with a
row_range.

```python
my_row_set = row_set.from_rows_or_ranges(row_range.infinite)
my_truncated_row_set = row_set.intersect(my_row_set, row_range.right_open("row-a", "row-h"))
```

## Specifying a version of a value

Bigtable lets you keep many values in one cell with different timestamps. You can specify which version you want to pick
using version filters. However, you can only retrieve a two dimensional vector using pytorch_bigtable connector, so
`latest` filter is always appended to the user specified version filter. Meaning, if more than one value for one cell
goes through the provided filter, the newer shall be used.

You can either use the `latest` filter passing the newest value, or you can specify a time range. The time range can be
provided either as python datetime objects or a number representing seconds or microseconds since epoch.

```python
import pytorch_bigtable.version_filters as version_filters

start = datetime(2020, 10, 10, 12, 0, 0)
end = datetime(2100, 10, 10, 13, 0, 0)
version_filters.timestamp_range(start, end)
version_filters.timestamp_range(int(start.timestamp()), int(end.timestamp()))
```

## Writing to Bigtable

To put data in Bigtable, you can use the write_tensor method. You have to provide row_keys for the data you're writing.
To do that, you can either pass a list of strings, or a callback i.e. a python function that will be called to generate
a row_key for each row with two arguments:

* a tensor representing a row that is currently being written of shape `[1,n]`
  where n is a number of columns
* an index of the current row

You can use that callback to avoid creating a very large list of row_keys. Remember that putting consecutive rows in
Bigtable is an anti-pattern and you should avoid that. Easiest option would be to provide a callback generating random
row_keys for each row.

**Note** This is by no means optimal or efficient way of sending the data to Bigtable. If you're looking for uploading
large quantities of data to Bigtable efficiently, please consider
using [BT client libraries](https://cloud.google.com/bigtable/docs/reference/libraries) which are designated for that.

```python
def row_callback(tensor, index):
  return "row" + str(random.randint(1000, 9999)).rjust(4, "0")


table.write_tensor(ten, ["fam1:col1", "fam2:col2"], row_callback)
```

