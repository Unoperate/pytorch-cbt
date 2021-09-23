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

# This example is based on data from:
# https://github.com/Fraud-Detection-Handbook/simulated-data

import os
import argparse

import torch.utils.data
from tqdm import tqdm
from sklearn.metrics import average_precision_score, precision_recall_curve
from matplotlib import pyplot as plt
import pytorch_bigtable as pbt
import pandas as pd

output_feature = "TX_FRAUD"

input_features = ['TX_AMOUNT', 'TX_DURING_WEEKEND', 'TX_DURING_NIGHT',
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


class FraudDataset(torch.utils.data.Dataset):
  def __init__(self, transactions_df):
    self.X = torch.tensor(transactions_df[input_features].values,
                          dtype=torch.float32)
    self.y = torch.tensor(transactions_df[output_feature].values,
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
  layer_size = len(input_features)
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
      if isinstance(loader.dataset, FraudDataset):
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


def eval_model(model, loader, output="precision_recall.png"):
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
    precision, recall, thresholds = precision_recall_curve(y_all, y_pred_all)
    plt.figure()
    plt.step(recall, precision, alpha=0.3, color='b')
    plt.fill_between(recall, precision, alpha=0.3, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(output, dpi=300, format='png')


if __name__ == "__main__":

  parser = argparse.ArgumentParser("fraud_example.py")
  parser.add_argument("-b", "--use_bigtable",
                      help="specifies if training data is taken from bigtable "
                           "database.",
                      action='store_true')
  args = parser.parse_args()

  if args.use_bigtable:
    print("connecting to BigTable")
    os.environ["BIGTABLE_EMULATOR_HOST"] = "172.17.0.1:8086"
    client = pbt.BigtableClient("unoperate-test", "172.17.0.1:8086",
                                endpoint="")

    train_table = client.get_table("train")

    print("creating train set")
    train_set = train_table.read_rows(torch.float32,
      ["cf1:" + column for column in input_features] + [
        "cf1:" + output_feature],
      pbt.row_set.from_rows_or_ranges(pbt.row_range.infinite()))
  else:
    print("creating train set")
    train_df = pd.read_csv("train_df.csv", parse_dates=['TX_DATETIME'])
    train_set = FraudDataset(train_df)

  print("creating test set")
  test_df = pd.read_csv("test_df.csv", parse_dates=['TX_DATETIME'])
  test_set = FraudDataset(test_df)

  print("creating a model")
  model = create_model()
  learning_rate = 2e-3
  batch_size = 10000

  train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

  test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

  print("initial testing")
  eval_model(model=model, loader=test_loader, output="before.png")

  print("training")
  train_model(model=model, loader=train_loader, optimizer=optimizer,
              loss_fn=torch.nn.BCELoss(), epochs=20)

  torch.save(model.state_dict(), "model.backup")
  # model.load_state_dict(torch.load("model.backup"))

  print("testing after training")
  eval_model(model=model, loader=test_loader, output="after.png")
