import torch
import pytorch_bigtable as pbt
import pandas as pd


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

def train_model(model, loader, optimizer, max_epochs=50):
  torch.manual_seed(1234567)
  loss_fn = torch.nn.BCELoss()
  model.train()
  model.apply(init_weights)
  total_loss = 0
  for epoch in range(1, max_epochs+1):
    for i, data in enumerate(loader):
      X, y = data[:,:-1], data[:,-1]
      y_pred = model(X)
      loss = loss_fn(y_pred.reshape(-1), y)
      total_loss += loss.item()
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      if i % 100 == 0:
        print('at iter {}, loss={}'.format(i, loss.item()))
    print('At epoch {}, loss={}'.format(epoch, total_loss))
  print('Final loss={}'.format(total_loss))
  return

def eval_model(model, eval_set, batch_size):
  model.eval()
  loader = torch.utils.data.DataLoader(eval_set, batch_size)
  correct, total = 0, 0
  with torch.no_grad():
    for i, data in enumerate(loader, 0):
      X, y = data
      y_pred = model(X)
      correct += (y_pred > 0.5 == y).sum()
      total += y.shape[0]

  print('test accuracy {}/{}  {}%'.format(correct, total, correct/total * 100))

if __name__ == '__main__':
  print("connecting to BigTable")
  client = pbt.BigtableClient("unoperate-test", "test-instance-id", endpoint="")
  train_table = client.get_table("train")

  print("creating train set")
  train_set = train_table.read_rows(
    torch.float32,
    ["cf1:" + column for column in input_features] + ["cf1:"+output_feature],
    pbt.row_set.from_rows_or_ranges(pbt.row_range.infinite()))


  print("creating test set")
  test_df = pd.read_csv("test_df.csv", parse_dates=['TX_DATETIME'])
  test_set = CreditCardDataset(test_df[input_features], test_df[output_feature])

  print("creating a model")
  model = create_model()
  learning_rate = 0.002
  batch_size=1000

  loader = torch.utils.data.DataLoader(train_set, batch_size)
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)

  print("training")
  train_model(model=model, loader=loader, optimizer=optimizer,  max_epochs=10)

  print("testing")
  eval_model(model=model, eval_set=test_set, batch_size=batch_size)


