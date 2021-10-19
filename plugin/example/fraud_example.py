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
#
# This is an example of training a NN for fraud detection using data
# stored in Google Cloud Bigtable.
#
# We use train data prepared in `seed_bigtable.py`. It can be either loaded
# as a CSV or downloaded directly from Google Cloud Bigtable.
#
# First, a fully connected Neural Network is initialized and tested on the
# test data loaded from the CSV. The metrics are average precision and a
# precision/recall curve stored in a file before.png.
#
# Next, we train the network using a pytorch dataset that connects to Bigtable.
#
# Last, we evaluate the network again, using the test dataset.
#
#
# This example is based on data from:
# https://github.com/Fraud-Detection-Handbook/simulated-data
# For more information please refer to comments in `seed_bigtable.py`.

import os
import argparse

import torch.utils.data
from tqdm import tqdm
from sklearn.metrics import average_precision_score, precision_recall_curve
from matplotlib import pyplot as plt
import pytorch_bigtable as pbt
import pytorch_bigtable.row_set
import pytorch_bigtable.row_range
import pandas as pd

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


class InMemFraudDataset(torch.utils.data.Dataset):
  def __init__(self, transactions_df):
    self.X = torch.tensor(transactions_df[INPUT_FEATURES].values,
                          dtype=torch.float32)
    self.y = torch.tensor(transactions_df[OUTPUT_FEATURE].values,
                          dtype=torch.float32)

  def __len__(self):
    return len(self.X)

  def __getitem__(self, item):
    return (self.X[item], self.y[item])


def init_weights(layer):
  if isinstance(layer, torch.nn.Linear):
    torch.nn.init.xavier_uniform_(layer.weight)
    layer.bias.data.fill_(0.01)


def create_model():
  layer_size = len(INPUT_FEATURES)
  model = torch.nn.Sequential(torch.nn.Linear(layer_size, layer_size),
                              torch.nn.ReLU(),
                              torch.nn.Linear(layer_size, layer_size),
                              torch.nn.ReLU(), torch.nn.Linear(layer_size, 1),
                              torch.nn.Sigmoid())
  return model


def train_model(model, loader, optimizer, loss_fn, epochs=20):
  model.apply(init_weights)
  for epoch in tqdm(range(epochs), ascii=True, desc="training"):
    total_loss = 0
    total_prec = 0
    batch_counter = 0
    for batch in loader:
      # if the data comes from Bigtable, it is in form of one vector,
      # not a tuple (features, label) so we need to split it ourselves.
      if isinstance(loader.dataset, InMemFraudDataset):
        X, y = batch
      else:
        X, y = batch[:, :-1], batch[:, -1]
      y_pred = model(X)
      loss = loss_fn(y_pred.reshape(-1), y)
      total_loss += loss.item()
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      total_prec += average_precision_score(y.long(), y_pred.detach())
      batch_counter += 1

    tqdm.write(f"epoch-{epoch}: loss={total_loss}, avg_precision="
               f"{total_prec / batch_counter :.4f}")


def eval_model(model, loader, curve_plot_path=None):
  model.eval()
  with torch.no_grad():
    y_all = None
    y_pred_all = None
    for batch in loader:
      X, y = batch
      y_pred = model(X)
      if y_all is None:
        y_all = y
        y_pred_all = y_pred
      else:
        y_all = torch.cat((y_all, y), 0)
        y_pred_all = torch.cat((y_pred_all, y_pred), 0)

    print(f"average precision: {average_precision_score(y, y_pred)}")
    if curve_plot_path:
      precision, recall, thresholds = precision_recall_curve(y_all, y_pred_all)
      plt.figure()
      plt.step(recall, precision, alpha=0.3, color='b')
      plt.fill_between(recall, precision, alpha=0.3, color='b')
      plt.xlabel('Recall')
      plt.ylabel('Precision')
      plt.savefig(curve_plot_path, dpi=300, format='png')


def main(args):

  if args.use_bigtable:
    print("connecting to BigTable")
    if args.emulator_host:
      os.environ["BIGTABLE_EMULATOR_HOST"] = args.emulator_host
    client = pbt.BigtableClient(args.project_id, args.instance_id)

    train_table = client.get_table(args.table_id)

    print("creating train set")
    train_set = train_table.read_rows(torch.float32,
                                      ["cf1:" + column for column in
                                       INPUT_FEATURES] + [
                                        "cf1:" + OUTPUT_FEATURE],
                                      pbt.row_set.from_rows_or_ranges(
                                        pbt.row_range.infinite()))
  else:
    print("creating train set")
    train_df = pd.read_csv(args.train_set, parse_dates=['TX_DATETIME'])
    train_set = InMemFraudDataset(train_df)

  print("creating test set")
  test_df = pd.read_csv(args.test_set, parse_dates=['TX_DATETIME'])
  test_set = InMemFraudDataset(test_df)

  print("creating a model")
  model = create_model()
  learning_rate = 2e-3
  batch_size = 10000

  train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                             num_workers=5)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

  print("initial testing")
  before_curve = "before.png" if args.draw_curves else None
  eval_model(model=model, loader=test_loader, curve_plot_path=before_curve)

  print("training")
  train_model(model=model, loader=train_loader, optimizer=optimizer,
              loss_fn=torch.nn.BCELoss(), epochs=20)

  print("testing after training")
  after_curve = "after.png" if args.draw_curves else None
  eval_model(model=model, loader=test_loader, curve_plot_path=after_curve)


def parse_arguments():
  parser = argparse.ArgumentParser("fraud_example.py")
  parser.add_argument("-b", "--use_bigtable",
                      help="specifies if training data is taken from bigtable "
                           "database.", action='store_true')
  parser.add_argument("-p", "--project_id",
                      help="google cloud bigtable project id")
  parser.add_argument("-i", "--instance_id",
                      help="google cloud bigtable instance id")
  parser.add_argument("-t", "--table_id", help="google cloud bigtable table id")
  parser.add_argument("-f", "--family",
                      help="column family that will be used for all the "
                           "columns")
  parser.add_argument("-e", "--emulator_host",
                      help="google cloud bigtable emulator host in format "
                           "ip:port")
  parser.add_argument("-r", "--train_set", help="path to train set CSV")
  parser.add_argument("-s", "--test_set", help="path to test set CSV",
                      required=True)
  parser.add_argument("-c", "--draw_curves",
                      help="Should the model evaluation draw precision-recall "
                           "curves and save them?",
                      action='store_true')

  args = parser.parse_args()

  if args.use_bigtable and (
      args.project_id is None or args.instance_id is None or args.table_id is
      None or args.family is None):
    parser.error(
      "--use_bigtable requires --project_id, --instance_id --table_id and "
      "--family")

  if args.use_bigtable is None and (args.train_set is None):
    parser.error(
      "if not connecting to Bigtable, path to train set must be specified by "
      "setting --train_set")

  return args

if __name__ == "__main__":

  args = parse_arguments()

  main(args)
