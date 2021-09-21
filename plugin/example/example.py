import os
import torch
import pytorch_bigtable as pbt
import pandas as pd
from sklearn.metrics import recall_score, roc_auc_score, balanced_accuracy_score,precision_recall_curve, average_precision_score
from matplotlib import pyplot as plt


class CreditCardDataset(torch.utils.data.Dataset):
  def __init__(self, X, y):
    self.X = torch.tensor(X.values, dtype=torch.float32)
    self.y = torch.tensor(y.values, dtype=torch.float32)

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return (self.X[idx], self.y[idx])

  def __str__(self):
    return 'CreditCardDataset has {} items and {} frauds'.format(len(self.y), int(self.y.sum()))


output_feature="TX_FRAUD"

input_features=['TX_AMOUNT','TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
                'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
                'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
                'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
                'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
                'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
                'TERMINAL_ID_RISK_30DAY_WINDOW']

def create_model():
  d_in, d_h1, d_h2, d_out = len(input_features), len(input_features), len(input_features), 1
  nn_model = torch.nn.Sequential(
    torch.nn.Linear(d_in, d_h1),
    torch.nn.ReLU(),
    torch.nn.Linear(d_h1, d_h2),
    torch.nn.ReLU(),
    torch.nn.Linear(d_h2, d_out),
    torch.nn.Sigmoid(),
  )
  return nn_model


def init_weights(m):
  if type(m) == torch.nn.Linear:
    torch.nn.init.xavier_uniform_(m.weight)
    m.bias.data.fill_(0.1)

def train_model_bt(model, loader, optimizer, max_epochs=50):
  torch.manual_seed(66543)
  loss_fn = torch.nn.BCELoss()
  model.train()
  model.apply(init_weights)
  for epoch in range(1, max_epochs+1):
    total_loss = 0
    for i, data in enumerate(loader):
      X, y = data[:,:-1], data[:,-1]
      print("training on batch ", y.reshape(-1).sum(), '/', y.shape[0])
      y_pred = model(X)
      loss = loss_fn(y_pred.reshape(-1), y)
      total_loss += loss.item()
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      if i % 10 == 0:
        print('at iter {}, loss={}'.format(i, loss.item()))
    print('At epoch {}, loss={}'.format(epoch, total_loss))
  return

def train_model(model, loader, optimizer, max_epochs=50):
  torch.manual_seed(66543)
  loss_fn = torch.nn.BCELoss()
  model.train()
  model.apply(init_weights)
  for epoch in range(1, max_epochs+1):
    total_loss = 0

    for i, data in enumerate(loader, 0):
      X, y = data
      # print("training on batch ", y.reshape(-1).sum(), '/', y.shape[0])
      y_pred = model(X)
      loss = loss_fn(y_pred.reshape(-1), y)
      total_loss += loss.item()
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      if i % 10 == 0:
        print('at iter {}, loss={}'.format(i, loss.item()))
    print('At epoch {}, loss={}'.format(epoch, total_loss))
  return

def eval_model(model, eval_set, batch_size, output="out.png"):
  model.eval()
  loader = torch.utils.data.DataLoader(eval_set, batch_size)
  with torch.no_grad():
    correct, total = 0, 0
    y_all = None
    y_pred_all = None
    for i, data in enumerate(loader, 0):
      X, y = data
      y_pred = model(X)
      if i == 0:
        y_all = y
        y_pred_all = y_pred
      else:
        y_all = torch.cat((y_all, y), 0)
        y_pred_all = torch.cat((y_pred_all, y_pred), 0)

    print("roc_auc score:", roc_auc_score(y_all, y_pred_all))
    print("recall score:", recall_score(y_all, (y_pred_all > 0.5).long() ))
    print("average precision score:", average_precision_score(y_all, y_pred_all))
    precision, recall, thresholds = precision_recall_curve(y_all, y_pred_all)
    plt.figure()
    plt.step(recall, precision, alpha=0.3, color='b')
    plt.fill_between(recall, precision, alpha=0.3, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(output, dpi=300, format='png')

if __name__ == '__main__':
  print("connecting to BigTable")
  os.environ["BIGTABLE_EMULATOR_HOST"] = "172.17.0.1:8086"
  client = pbt.BigtableClient("unoperate-test", "172.17.0.1:8086", endpoint="")

  train_table = client.get_table("train")

  print("creating train set")
  train_set = train_table.read_rows(
    torch.float32,
    ["cf1:" + column for column in input_features] + ["cf1:"+output_feature],
    pbt.row_set.from_rows_or_ranges(pbt.row_range.infinite()))

  # print("creating train set")
  # train_df = pd.read_csv("train_df.csv", parse_dates=['TX_DATETIME'])
  # train_set = CreditCardDataset(train_df[input_features], train_df[output_feature])

  print("creating test set")
  test_df = pd.read_csv("test_df.csv", parse_dates=['TX_DATETIME'])
  test_set = CreditCardDataset(test_df[input_features], test_df[output_feature])

  print("creating a model")
  model = create_model()
  learning_rate = 2e-3
  batch_size=10000

  loader = torch.utils.data.DataLoader(train_set, batch_size)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


  print("initial testing")
  eval_model(model=model, eval_set=test_set, batch_size=batch_size, output="before.png")

  print("training")
  # train_model(model=model, loader=loader, optimizer=optimizer,  max_epochs=20)
  train_model_bt(model=model, loader=loader, optimizer=optimizer,  max_epochs=10)

  torch.save(model.state_dict(), "model.backup")
  model.load_state_dict(torch.load("model.backup"))



  print("testing after training")
  eval_model(model=model, eval_set=test_set, batch_size=batch_size, output="after.png")


