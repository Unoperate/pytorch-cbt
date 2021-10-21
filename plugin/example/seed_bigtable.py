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
#
# We provide an example of training a NN for fraud detection using data
# stored in BigTable in `fraud_example.py`. To showcase what our plugin does,
# we focus on **reading** the data that is already in Google Cloud Bigtable,
# while training the network.
#
# The code in this file is used to upload the data and is by no means
# optimal or efficient and SHOULD NOT BE USED for any real application. If
# you're looking for uploading data to BT efficiently, please consider
# using BT client libraries:
# https://cloud.google.com/bigtable/docs/reference/libraries
#
# Using the code from the Fraud Detection Handbook that is available freely
# on github you can generate as big of a dataset as you wish. However,
# for the sake of this example, we use the pre-generated and pre-formatted
# data that was provided in the repository. This code loads that data into
# memory, splits it into training and testing sets, saves both as CSV and
# uploads the training set to BT.
#
# This example is based on data from:
# https://github.com/Fraud-Detection-Handbook/simulated-data

import random
from datetime import datetime, timedelta
import os
import pickle
import pandas as pd
from typing import Union
import pytorch_bigtable as pbt
import torch
import argparse
from tqdm import tqdm
import requests
import zipfile
import io
import tempfile
import glob

OUTPUT_FEATURE = "TX_FRAUD"

INPUT_FEATURES = ['TX_AMOUNT', 'TX_DURING_WEEKEND', 'TX_DURING_NIGHT',
                  'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
                  'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW',
                  'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
                  'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW',
                  'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
                  'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW',
                  'TERMINAL_ID_NB_TX_1DAY_WINDOW',
                  'TERMINAL_ID_RISK_1DAY_WINDOW',
                  'TERMINAL_ID_NB_TX_7DAY_WINDOW',
                  'TERMINAL_ID_RISK_7DAY_WINDOW',
                  'TERMINAL_ID_NB_TX_30DAY_WINDOW',
                  'TERMINAL_ID_RISK_30DAY_WINDOW']


def ensure_datetime(value: Union[str, datetime]):
  """Makes sure that the value is in the right format and return a datetime
  object. Raises TypeError if format is wrong."""
  if isinstance(value, datetime):
    return value
  try:
    parsed_date = datetime.strptime(value, "%Y-%m-%d")
  except:
    raise TypeError(f"{value} must be in format %Y-%m-%d")
  return parsed_date


def read_data(path_to_repository: str,
              start: Union[str, datetime] = "2018-04-01",
              end: Union[str, datetime] = "2018-10-01"):
  """Function for reading transactions dataset taken from
  https://github.com/Fraud-Detection-Handbook/simulated-data

  Args:
    start (str|datetime): start of range (inclusive) of days that should be
    used to construct the dataset. Either a python datetime or a string in
    format "%Y-%m-%d". The default value is the first day in the
    pre-generated dataset.
    end (str|datetime): end of range (exclusive), type same as `start`.
  """
  data_dir = os.path.join(path_to_repository,
                          'simulated-data-transformed-6e3ca5849b4681430388056d3f1dcfb41d4e8269',
                          'data')

  start_date = ensure_datetime(start)
  end_date = ensure_datetime(end)

  all_dfs = []
  for f in os.listdir(data_dir):
    date = datetime.strptime(f, "%Y-%m-%d.pkl")
    if start_date <= date < end_date:
      with open(os.path.join(data_dir, f), "rb") as file:
        all_dfs.append(pickle.load(file))

  output = pd.concat(all_dfs).replace([-1], 0)
  output.sort_values('TRANSACTION_ID').reset_index(drop=True, inplace=True)
  return output


def train_test_split(transactions_df: pd.DataFrame, days_train: int = 150,
                     delay: int = 15, days_test: int = 18):
  """Function splitting the transactions into train and test datasets.

  We put the transactions in two sets, transactions taking place in the first
  `days_train` days, representing the training period, and the transactions
  after that, representing testing period. We also create a delay between the
  two and exclude from the test set transactions made during the training
  period by customers known to be fraudulent.

  Args:
    transactions_df (DataFrame): dataframe with transactions.
    days_train (int): length of the training period in days.
    delay (int): length of the minimal delay in days between last transaction
    in train set and first in test set.
    days_test: length of the testing period in days.
  """
  train_date_start = transactions_df['TX_DATETIME'].min().replace(hour=0,
                                                                  minute=0,
                                                                  second=0)
  train_date_end = train_date_start + timedelta(days=days_train)
  test_date_start = train_date_end + timedelta(days=delay)
  test_date_end = test_date_start + timedelta(days=days_test)

  train_df = transactions_df[transactions_df['TX_DATETIME'] < train_date_end]

  fraudulent_customer_ids = transactions_df[
    (transactions_df['TX_DATETIME'] < test_date_start) & (
        transactions_df['TX_FRAUD'] == 1)]['CUSTOMER_ID'].unique()
  test_df = transactions_df[
    (transactions_df['TX_DATETIME'] >= test_date_start) & (
        transactions_df['TX_DATETIME'] < test_date_end) & (
      ~transactions_df['CUSTOMER_ID'].isin(fraudulent_customer_ids))]

  return train_df, test_df


def parse_arguments():
  parser = argparse.ArgumentParser("seed_bigtable.py",
                                   description="Script uploading fraud "
                                               "detection dataset to "
                                               "bigtable. For more "
                                               "information please visit: "
                                               "https://github.com/Unoperate/pytorch-cbt")
  parser.add_argument("-p", "--project_id",
                      help="google cloud bigtable project id", required=True)
  parser.add_argument("-i", "--instance_id",
                      help="google cloud bigtable instance id", required=True)
  parser.add_argument("--train_set_table",
                      help="google cloud bigtable table storing train_set",
                      required=True)
  parser.add_argument("--test_set_table",
                      help="google cloud bigtable table storing test_set",
                      required=True)
  parser.add_argument("-f", "--family",
                      help="column family that will be used for all the "
                           "columns", required=True)
  parser.add_argument("-e", "--emulator_addr",
                      help="google cloud bigtable emulator address in format "
                           "ip:port")

  return parser.parse_args()


def main(args):
  global_temp_dir = tempfile.gettempdir()
  temp_dirs = glob.glob(
    os.path.join(tempfile.gettempdir(), '*-simulated-data-transformed'))
  if len(temp_dirs) == 0:
    temp_dir = tempfile.mkdtemp(suffix="-simulated-data-transformed")
    print("downloading simulated-data-transformed to:", temp_dir)
    response = requests.get(
      "https://github.com/Fraud-Detection-Handbook/simulated-data-transformed"
      "/archive/6e3ca58.zip")
    with zipfile.ZipFile(io.BytesIO(response.content)) as simulated_data_zip:
      simulated_data_zip.extractall(temp_dir)
  elif len(temp_dirs) > 1:
    temp_dir = temp_dirs[0]
    print(
      "found more than one repositories with simulated-data-transformed, "
      "using:",
      temp_dir)
  else:
    temp_dir = temp_dirs[0]
    print("found simulated-data-transformed in:", temp_dir)

  transactions_df = read_data(temp_dir, end="2018-04-30")
  print("read", transactions_df.shape, 'transactions')
  train_df, test_df = train_test_split(transactions_df, days_train=20, delay=5,
                                       days_test=5)
  print("train set", train_df.shape, "containing", train_df['TX_FRAUD'].sum(),
        "fraudulent transactions")
  print("test set", test_df.shape, "containing", test_df['TX_FRAUD'].sum(),
        "fraudulent transactions")

  print("Seed bigtable")
  if args.emulator_addr:
    os.environ["BIGTABLE_EMULATOR_HOST"] = args.emulator_addr

  client = pbt.BigtableClient(args.project_id, args.instance_id)
  train_table = client.get_table(args.train_set_table)
  test_table = client.get_table(args.test_set_table)

  BATCH_SIZE = 1000

  X_train = torch.tensor(train_df[INPUT_FEATURES].values, dtype=torch.float32)
  y_train = torch.tensor(train_df[[OUTPUT_FEATURE]].values, dtype=torch.float32)

  row_keys_train = ["r" + str(j).rjust(10, '0') for j in
                    range(X_train.shape[0])]
  random.shuffle(row_keys_train)

  print("send train_set data")
  for i, idx in enumerate(
      tqdm(range(0, X_train.shape[0], BATCH_SIZE), ascii=True,
           desc="seeding BigTable")):
    batch_X = X_train[idx:idx + BATCH_SIZE]
    batch_y = y_train[idx:idx + BATCH_SIZE]
    row_keys = row_keys_train[idx:idx + BATCH_SIZE]
    train_table.write_tensor(batch_X, [args.family + ":" + column for column in
                                       INPUT_FEATURES], row_keys)
    train_table.write_tensor(batch_y, [args.family + ":" + OUTPUT_FEATURE],
                             row_keys)

  X_test = torch.tensor(test_df[INPUT_FEATURES].values, dtype=torch.float32)
  y_test = torch.tensor(test_df[[OUTPUT_FEATURE]].values, dtype=torch.float32)

  row_keys_test = ["r" + str(j).rjust(10, '0') for j in range(X_test.shape[0])]
  random.shuffle(row_keys_test)

  print("send test_set data")
  for i, idx in enumerate(
      tqdm(range(0, X_test.shape[0], BATCH_SIZE), ascii=True,
           desc="seeding BigTable")):
    batch_X = X_test[idx:idx + BATCH_SIZE]
    batch_y = y_test[idx:idx + BATCH_SIZE]
    row_keys = row_keys_test[idx:idx + BATCH_SIZE]
    test_table.write_tensor(batch_X, [args.family + ":" + column for column in
                                      INPUT_FEATURES], row_keys)
    test_table.write_tensor(batch_y, [args.family + ":" + OUTPUT_FEATURE],
                            row_keys)


if __name__ == '__main__':
  args = parse_arguments()
  main(args)
