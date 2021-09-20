import datetime
import os
import random

import pandas as pd
import torch
import pytorch_bigtable as pbt


if not os.path.exists("simulated-data-transformed"):
  os.system("git clone https://github.com/Fraud-Detection-Handbook/simulated-data-transformed")


# Load a set of pickle files, put them together in a single DataFrame, and order them by time
# It takes as input the folder DIR_INPUT where the files are stored, and the BEGIN_DATE and END_DATE
def read_from_files(DIR_INPUT, BEGIN_DATE, END_DATE):

  files = [os.path.join(DIR_INPUT, f) for f in os.listdir(DIR_INPUT) if f>=BEGIN_DATE+'.pkl' and f<=END_DATE+'.pkl']

  frames = []
  for f in files:
    df = pd.read_pickle(f)
    frames.append(df)
    del df
  df_final = pd.concat(frames)

  df_final=df_final.sort_values('TRANSACTION_ID')
  df_final.reset_index(drop=True,inplace=True)
  #  Note: -1 are missing values for real world data
  df_final=df_final.replace([-1],0)

  return df_final


def get_train_test_set(transactions_df,
                       start_date_training,
                       delta_train=7,delta_delay=7,delta_test=7):

  # Get the training set data
  train_df = transactions_df[(transactions_df.TX_DATETIME>=start_date_training) &
                             (transactions_df.TX_DATETIME<start_date_training+datetime.timedelta(days=delta_train))]

  # Get the test set data
  test_df = []

  # Note: Cards known to be frauded after the delay period are removed from the test set
  # That is, for each test day, all frauds known at (test_day-delay_period) are removed

  # First, get known frauded customers from the training set
  known_frauded_customers = set(train_df[train_df.TX_FRAUD==1].CUSTOMER_ID)

  # Get the relative starting day of training set (easier than TX_DATETIME to collect test data)
  start_tx_time_days_training = train_df.TX_TIME_DAYS.min()

  # Then, for each day of the test set
  for day in range(delta_test):

    # Get test data for that day
    test_df_day = transactions_df[transactions_df.TX_TIME_DAYS==start_tx_time_days_training+
                                  delta_train+delta_delay+
                                  day]

    # Frauded cards from that test day, minus the delay period, are added to the pool of known frauded customers
    test_df_day_delay_period = transactions_df[transactions_df.TX_TIME_DAYS==start_tx_time_days_training+
                                               delta_train+
                                               day-1]

    new_frauded_customers = set(test_df_day_delay_period[test_df_day_delay_period.TX_FRAUD==1].CUSTOMER_ID)
    known_frauded_customers = known_frauded_customers.union(new_frauded_customers)

    test_df_day = test_df_day[~test_df_day.CUSTOMER_ID.isin(known_frauded_customers)]

    test_df.append(test_df_day)

  test_df = pd.concat(test_df)

  # Sort data sets by ascending order of transaction ID
  train_df=train_df.sort_values('TRANSACTION_ID')
  test_df=test_df.sort_values('TRANSACTION_ID')

  return (train_df, test_df)


DIR_INPUT='./simulated-data-transformed/data/'

BEGIN_DATE = "2018-04-01"
END_DATE = "2018-09-30"
TEST_DATA_FILE = "test_df.csv"

output_feature="TX_FRAUD"

input_features=['TX_AMOUNT','TX_DURING_WEEKEND', 'TX_DURING_NIGHT', 'CUSTOMER_ID_NB_TX_1DAY_WINDOW',
                'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
                'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
                'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
                'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
                'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
                'TERMINAL_ID_RISK_30DAY_WINDOW']

print("Load  files")
transactions_df=read_from_files(DIR_INPUT, BEGIN_DATE, END_DATE)
print("{0} transactions loaded, containing {1} fraudulent transactions".format(len(transactions_df),transactions_df.TX_FRAUD.sum()))

print("Split dataset")
start_date_training = datetime.datetime.strptime("2018-04-01", "%Y-%m-%d")

(train_df, test_df)=get_train_test_set(transactions_df,start_date_training,
                                       delta_train=150,delta_delay=16,delta_test=16)


print("training set contains {0} transactions, {1} fraudulent".format(len(train_df),train_df.TX_FRAUD.sum()))
print("train", train_df.shape)
print("test set contains {0} transactions, {1} fraudulent".format(len(test_df),test_df.TX_FRAUD.sum()))
print("test", test_df.shape)

print("Seed bigtable")
client = pbt.BigtableClient("unoperate-test", "test-instance-id", endpoint="")
train_table = client.get_table("train")

BATCH_SIZE = 1000

X_train =  torch.tensor(train_df[input_features].values, dtype=torch.float32)
y_train =  torch.tensor(train_df[output_feature].values, dtype=torch.float32)

row_keys_all = ["r" + str(j).rjust(10, '0') for j in range(X_train.shape[0])]
random.shuffle(row_keys_all)

print("seeding train table...")
for i,idx in enumerate(range(0,X_train.shape[0],BATCH_SIZE)):
  batch_x = X_train[idx:idx+BATCH_SIZE]
  batch_y = y_train[idx:idx+BATCH_SIZE]
  row_keys = row_keys_all[idx:idx+BATCH_SIZE]
  print("done", i, "/", X_train.shape[0]//BATCH_SIZE)

  train_table.write_tensor(batch_x, ["cf1:" + column for column in input_features], row_keys)
  train_table.write_tensor(batch_y.reshape(-1,1), ["cf1:"+output_feature], row_keys)


print("saving test data locally")
test_df.to_csv('test_df.csv', index=False)
