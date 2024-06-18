import torch
import torch.nn as nn

import pandas as pd
import sklearn
import sklearn.model_selection

from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import one_hot
def prepare_data():
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower',
                    'Weight', 'Acceleration', 'Model year', 'Origin' ]
 
    # read_csv, at the same time assign the header   
    df = pd.read_csv('./auto+mpg/auto-mpg.data', names = column_names,
                     na_values="?", comment='\t', sep=" ",
                     skipinitialspace=True)

    df = df.dropna() # drop not a number 
    df = df.reset_index(drop=True) # reset the indices after drop na

    df_train, df_test = sklearn.model_selection.train_test_split(
            df, train_size=0.8, random_state=1)

    train_stats = df_train.describe().transpose()

    numeric_column_names = [
            'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration'
        ]

    df_train_norm, df_test_norm = df_train.copy(), df_test.copy()

    # normalize data
    for col_name in numeric_column_names:
        mean = train_stats.loc[col_name, 'mean'] # locate
        std  = train_stats.loc[col_name, 'std']
 
        # need to change the data type       
        df_train_norm[col_name] = df_train_norm[col_name].astype('float64') 
        df_test_norm[col_name] = df_test_norm[col_name].astype('float64')
        df_train_norm.loc[:, col_name] = (df_train_norm.loc[:, col_name] - mean)/std
        df_test_norm.loc[:, col_name] = (df_test_norm.loc[:, col_name] - mean)/std

    boundaries = torch.tensor([73,76,79])
    v = torch.tensor(df_train_norm['Model year'].values) # df.values is numpy array
    
    df_train_norm['Model year bucketed'] = torch.bucketize(
            v, boundaries, right = True
        )
    v = torch.tensor(df_test_norm['Model year'].values)

    df_test_norm['Model year bucketed'] = torch.bucketize(
            v, boundaries, right = True
        )

    numeric_column_names.append('Model year bucketed')
    total_origin = len(set(df_train_norm['Origin']))

    origin_encoded = one_hot(
            torch.from_numpy(df_train_norm['Origin'].values) % total_origin
        )
    x_train_numeric = torch.tensor(df_train_norm[numeric_column_names].values)
    x_train = torch.cat([x_train_numeric, origin_encoded], 1).float()

    origin_encoded = one_hot(
            torch.from_numpy(df_test_norm['Origin'].values) % total_origin
        )
    x_test_numeric = torch.tensor(df_test_norm[numeric_column_names].values)
    x_test = torch.cat([x_test_numeric, origin_encoded], 1).float()

    y_train = torch.tensor(df_train_norm['MPG'].values).float()
    y_test = torch.tensor(df_test_norm['MPG'].values).float()

    return x_train, y_train, x_test, y_test

def train_nn(data):
    x_train, y_train, x_test, y_test = data
    train_ds = TensorDataset(x_train, y_train)
    batch_size = 8
    torch.manual_seed(1)

    train_dl = DataLoader(train_ds, batch_size, shuffle=True)

    hidden_units = [8,4]
    all_layers = []
    input_size = x_train.shape[1]
    for hidden_unit in hidden_units:
        layer = nn.Linear(input_size, hidden_unit)
        input_size = hidden_unit
        all_layers.append(layer)
        all_layers.append(nn.ReLU())

    # mpg is a function of variable input,  mpg is the single output
    all_layers.append(nn.Linear(hidden_units[-1],1))
    model = nn.Sequential(*all_layers)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    num_epochs = 200
    log_epochs = 20

    for epoch in range(num_epochs):
        loss_hist_train = 0
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)[:,0]
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist_train += loss.item()

        if epoch % log_epochs == 0:
            print(f'epoch {epoch} Loss {loss_hist_train/len(train_dl):.4f}')

data =  prepare_data()
train_nn(data)
