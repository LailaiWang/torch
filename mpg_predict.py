import torch

def prepare_data():
    import pandas as pd
    url = 'http://archive.ics.uci.edu/ml/machining-learning-databses/auto-mpg/auto-mpg.data'
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower',
                    'Weight', 'Acceleration', 'Model year', 'Origin'
                   ]

    df = pd.read_csv(url, names = column_names, na_values="?", comment='\t', sep=" ",
                     skipinitialspace=True
                    )

    df = df.dropna()
    df = df.reset_index(drop=True)

    import sklearn
    import sklearn.model_selection

    df_train, df_test = sklearn.model_selection.train_test_split(
            df, train_size=0.8, random_state=1
        )

    train_stats = df_train.describe().transpose()

    numeric_column_names = [
            'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration'
        ]

    df_train_norm, df_test_norm = df_train.copy(), df_test.copy()

    # normalize data
    for col_name in numeric_column_names:
        mean = train_stats.loc[col_name, 'mean']
        std  = train_stats.loc[col_name, 'std']

        df_train_norm.loc[:, col_name] = (df_train_norm.loc[:, col_name] - mean)/std
        df_test_norm.loc[:, col_name] = (df_test_norm.loc[:, col_name] - mean)/std

    boundaries = torch.tensor([73,76,79])
    v = torch.tensor(df_train_norm['Model year'].values)
    
    df_train_norm['Model year bucketed'] = torch.bucketize(
            v, boundaries, right = True
        )
    v = torch.tensor(df_test_norm['Model year'].values)

    df_test_norm['Model year bucketed'] = torch.bucketize(
            v, boundaries, right = True
        )

    numeric_columes.append('Model year bucketed')
    
    # use one-hot-encoding for categorical data

    from torch.nn.functional import one_hot
    total_origin = len(set(df_train_norm['Origin']))

    origin_encoded = one_hot(
            torch.from_numpy(df_train_norm_norm['Origin'].values) % total_origin
        )
    x_train_numeric = torch.tensor(df_train_norm[numeric_column_names].names)
    x_train = torch.cat([x_train_numeric, origin_encoded], 1).float()

    origin_encoded = one_hot(
            torch.from_numpy(df_test_norm_norm['Origin'].values) % total_origin
        )
    x_test_numeric = torch.tensor(df_test_norm[numeric_column_names].names)
    x_test = torch.cat([x_test_numeric, origin_encoded], 1).float()

    y_train = torch.tensor(df_train_norm['MPG'].values).float()
    y_test = torch.tensor(df_test_norm['MPG'].values).float()

    return x_train, y_train, x_test, y_test

def setup_nn(data):
    x_train, y_train, x_test, y_test = data
    train_ds = TensorDataset(x_train, y_train)
    batch_size = 8
    torch.manual_seed(1)
    train_dl = DataLoad(train_ds, batch_size, shuffle=True)


    hidden_units = [8,4]

    input_size = x_train.shape[1]
    all_layers = []
    
    for hidden_unit in hidden_units:
        layer = nn.Linear(input_size, hidden_unit)
        input_size = hidden_unit
        all_layers.append(layer)
        all_layers.append(nn.ReLU)

    all_layers.append(nn.Linear(hidden_units[-1],1))
    model = nn.Sequential(*all_layers)

    

