<!-- 
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
-->

# Pytorch Bigtable Extension

This is a Pytorch Extension used to connect to Google Cloud Bigtable.

### Contents:

* Installation
* Credentials
* Quickstart
* Parallel read
* Specifying a version of a value
* Specifying a version of a value
* Writing to Bigtable
* Building it locally
* Byte representation
* Example

## Installation

Make sure you have torch installed. Then just use pip to install the latest
version

```
pip install -i https://test.pypi.org/simple/ pytorch-bigtable
```

## Credentials

Right now only the default credentials are supported. To connect to Bigtable you
need to set the environment variable `GOOGLE_APPLICATION_CREDENTIALS`.
Replace `[PATH]` with the file path of the JSON file that contains your service
account key.

```python
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "[PATH]"
```

**Note**: If you're using the emulator, 
remmember to set the `BIGTABLE_EMULATOR_HOST` environment variable
as described [here](https://cloud.google.com/bigtable/docs/emulator).


## Quickstart

First you need to create a client and a table you would like to read from.

```python
import torch
import pytorch_bigtable as pbt
import random

# replace the project_id, instance_id and the name of the table with suitable values.
client = pbt.BigtableClient(project_id="test-project", instance_id="test-instance")
train_table = client.get_table("train")
```

Now we will write some data into Bigtable. To do that, we create a
tensor `data_tensor`. We provide a list of column names in
format `column_family:column_name` and a list of rowkeys.

```python
data_tensor = torch.Tensor(list(range(40))).reshape(20, 2)
random_row_keys = ["row" + str(random.randint(0, 999)).rjust(3, "0") for _ in range(20)]
train_table.write_tensor(data_tensor, ["cf1:col1", "cf1:col2"], random_row_keys)
```

Great! Now we can create a pytorch dataset that will read the data from our
table. To do that, you have to provide the type of the data you wish to read,
list of column names in format `column_family:column_name`, and a row_set that
you would like to read.

Keep in mind that that bigtable reads values in lexicographical order, 
not the order they were put in. We gave them random row-keys 
so they will be shuffled.

```python
row_set = pbt.row_set.from_rows_or_ranges(pbt.row_range.infinite())

train_dataset = train_table.read_rows(torch.float32, ["cf1:col1", "cf1:col2"], row_set)

for tensor in train_dataset:
  print(tensor)
```

That's it! Congrats!
You can also explore our example of training a fraud-detection model on data
from Bigtable in example.py

## Parallel read

Our dataset supports reading in parallel from Bigtable. To do that, create a
pytorch DataLoader and set num_workers to a number higher than one. When a Bigtable table instance is created, a list of tablets is fetched from bigquery. When pytorch's dataloader spawns workers, each worker computes it's share of work based on the tablets in the table and starts reading from their share of
tablets. 

Batching is also supported. You have to set the batch_size when constructing the data_loader as you would normally do with any other dataset.

**Note**: Keep in mind that when reading in parallel, the rows are not
guaranteed to be read in any particular order.

```python
train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=5, batch_size=10)
for tensor in train_loader:
  print(tensor)
```

## Reading specific row_keys

To read the data from Bigtable, you can specify a set of rows or a range or a
combination of those. We partly expose the C++ Bigtable Client api for that
purpose.

pytorch_bigtable.BigtableTable.read_rows method expects you to provide a
row_set. You can construct a row_set from row_keys or row_ranges as follows:

```python
row_range_below_300 = pbt.row_range.right_open("row000", "row300")

my_row_set = pbt.row_set.from_rows_or_ranges(row_range_below_300, "row585", "row832")
```

such row_set would contain a range of rows `[row000, row300)` and rows row585 and row832.

you can also create a row_set from an infinite range, empty range or a prefix.
You can also intersect it with a row_range.

```python
my_truncated_row_set = pbt.row_set.intersect(my_row_set, pbt.row_range.right_open("row200", "row700"))
```

## Specifying a version of a value

Bigtable lets you keep many values in one cell with different timestamps. You
can specify which version you want to pick using version filters. However, you
can only retrieve a two dimensional vector using pytorch_bigtable connector, so
`latest` filter is always appended to the user specified version filter.
Meaning, if more than one value for one cell goes through the provided filter,
the newer shall be used.

You can either use the `latest` filter passing the newest value, or you can
specify a time range. The time range can be provided either as python datetime
objects or a number representing seconds or microseconds since epoch.

```python
import pytorch_bigtable.version_filters as version_filters
from datetime import datetime, timezone

start = datetime(2020, 10, 10, 12, 0, 0, tzinfo=timezone.utc)
end = datetime(2100, 10, 10, 13, 0, 0, tzinfo=timezone.utc)
from_datetime = version_filters.timestamp_range(start, end)
from_posix_timestamp = version_filters.timestamp_range(int(start.timestamp()), int(end.timestamp()))
```

## Writing to Bigtable

To put data in Bigtable, you can use the write_tensor method. You have to
provide row_keys for the data you're writing. To do that, you can either pass a
list of strings, or a callback i.e. a python function that will be called to
generate a row_key for each row with two arguments:

* a tensor representing a row that is currently being written of shape `[1,n]`
  where n is a number of columns
* an index of the current row

You can use that callback to avoid creating a very large list of row_keys.
Remember that putting consecutive rows in Bigtable is an anti-pattern and you
should avoid that. Easiest option would be to provide a callback generating
random row_keys for each row.

**Note** This is by no means optimal or efficient way of sending the data to
Bigtable. If you're looking for uploading large quantities of data to Bigtable
efficiently, please consider
using [BT client libraries](https://cloud.google.com/bigtable/docs/reference/libraries)
which are designated for that.

```python
def row_callback(tensor, index):
  return "row" + str(random.randint(1000, 9999)).rjust(4, "0")


table.write_tensor(data_tensor, ["cf1:col1", "cf1:col2"], row_callback)
```

## Byte representation

Because the byte representation of variables differ depending on the
architecture of the machine the code is run on, we are using the xdr library to
convert the values to bytes. XDR is a part of rpc library. 

## Example

We provide a simple end-to-end example consisting of two files: 
`plugin/example/seed_bigtable.py` and `plugin/example/fraud_example.py`.

### seed_bigtable.py
It is used to generate credit-card transactions data as described in 
[Fraud-Detection-Handbook](https://github.com/Fraud-Detection-Handbook/simulated-data).
First some transactions are generated and stored in memory as a whole. Then they
are split to two datasets - train and test and uploaded to Bigtable. 


You have to specify the project and instance, the name of train and test table 
as well as column family which should be used for all the columns as 
script arguments. 


If you wish to use the emulator, provide the emulator address and port 
as an argument as well.


command to seed the database:
```bash
python3 seed_bigtable.py \
  --project_id test-project \
  --instance_id test-instance \
  --train_set_table train \
  --test_set_table test \
  -e "127.0.0.1:8086" \
  -f cf1
```

### fraud_example.py
It trains a simple fully-connected neural network for fraud detection based on
data taken straight from bigtable. Keep in mind that the dataset is synthetic
and the purpose of this example is to showcase the bigtable dataset and not
fraud-detection algorithm.

The network is first evaluated on the data from the `test` table, then the network
is trained and evaluated again to verify that there was in fact some improvement.


command to run the example:
```bash
python3 fraud_example.py  \
--project_id test-project \
--instance_id test-instance \
--train_set_table train \
--test_set_table test \
-e "127.0.0.1:8086" \
-f cf1
```
