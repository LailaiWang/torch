from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from torch.optim import SGD
import matplotlib.pyplot as plt
import numpy as np

def create_data(seed = 1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = np.random.uniform(low=-1, high=1, size=(200,2))
    y = np.ones(len(x))
    y[x[:,0]*x[:,1]<0] = 0

    n_train = 100
    x_train = torch.tensor(x[:n_train, :], dtype = torch.float32) 
    y_train = torch.tensor(y[:n_train], dtype = torch.float32) 
    x_valid = torch.tensor(x[n_train:, :], dtype = torch.float32) 
    y_valid = torch.tensor(y[n_train:], dtype = torch.float32) 

    train_ds = TensorDataset(x_train, y_train)
    batch_size = 2
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    return train_dl, x_valid, y_valid, n_train, batch_size

def train(model_input, num_epochs):

    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs

    # retrieve data first
    train_dl, x_valid, y_valid, n_train, batch_size = model_input.data 
    
    # start training
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_dl:
            pred = model_input.model(x_batch)[:,0]
            loss = model_input.loss(pred, y_batch)
            loss.backward() # backward differentiation
            model_input.optimizer.step() # iteration 
            model_input.optimizer.zero_grad() # reset gradient
            loss_hist_train[epoch] += loss.item() # retrive the value of loss function
            is_correct = ((pred>=0.5).float() == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.mean()

        loss_hist_train[epoch] /= n_train/batch_size # everage per batch
        accuracy_hist_train[epoch] /= n_train/batch_size
        pred = model_input.model(x_valid)[:,0]
        loss = model_input.loss(pred, y_valid)
        loss_hist_valid[epoch] = loss.item()
        is_correct = ((pred>=0.5).float() == y_valid).float()
        accuracy_hist_valid[epoch] += is_correct.mean()

    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid
 
def visualize_results(history, name):
    fig = plt.figure(figsize=(16,4))
    ax = fig.add_subplot(1,2,1)
    plt.plot(history[0], lw=4)
    plt.plot(history[1], lw=4)
    plt.legend(['Train loss', 'validation loss'], fontsize = 15)
    ax.set_xlabel('epoch', size=15)

    ax = fig.add_subplot(1,2,2)
    plt.plot(history[2], lw=4)
    plt.plot(history[3], lw=4)
    plt.legend(['Train acc', 'validation acc'], fontsize = 15)
    ax.set_xlabel('epoch', size=15)
    plt.savefig(f'{name}.png')

class Base(object):
    '''
    object new class method resolution order
    '''
    def __init__(self, data, x_valid, y_valid, n_train, n_batch, nin=2, nout=1, lr = 0.001):
        super(Base, self).__init__()
        self._ntrain = n_train
        self._nbatch = n_batch
        self._data = data
        self._valid = [x_valid, y_valid]
        self._lr = lr
        self._loss = nn.BCELoss()
        self._optimizer = None # torch.optim.SGD(self.model.parameters(), lr = self._lr)
        self._model = None

    @property
    def model(self):
        return self._model

    @property
    def loss(self):
        return self._loss
    
    @property
    def optimizer(self):
        return self._optimizer

    @property
    def data(self):
        return self._data, self._valid[0], self._valid[1], self._ntrain, self._nbatch
    
class NoHidden1(Base):
    def __init__(self, data, x_valid, y_valid, n_train, n_batch, nin=2, nout=1, lr = 0.001):
        super(NoHidden1, self).__init__(
            data, x_valid, y_valid, n_train, n_batch, nin, nout, lr
        )
        self._model = nn.Sequential(nn.Linear(nin,nout), nn.Sigmoid())
        self._optimizer = torch.optim.SGD(self.model.parameters(), lr = lr)

class TwoHiddenLayers1(Base):
    def __init__(self, data, x_valid, y_valid, n_train, n_batch, nin=2, nout=1, lr = 0.001):
        super(TwoHiddenLayers1, self).__init__(
            data, x_valid, y_valid, n_train, n_batch, nin, nout, lr
        )
        self._model = nn.Sequential(
                        nn.Linear(nin,4),
                        nn.ReLU(),
                        nn.Linear(4,4),
                        nn.ReLU(),
                        nn.Linear(4,nout),
                        nn.Sigmoid()
                      )
        self._optimizer = torch.optim.SGD(self.model.parameters(), lr = lr)

class NoisyLinear(nn.Module):
    def __init__(self, nin=2, nout=1, nstd=0.1):
        super(NoisyLinear, self).__init__()
        w = torch.Tensor(nin, nout)
        self.w= nn.Parameter(w)
        nn.init.xavier_uniform_(self.w)
        b=torch.Tensor(nout).fill_(0)
        self.b = nn.Parameter(b)
        self.noise_stddev = nstd

    def forward(self, x, training=False):
        if training:
            noise = torch.normal(0.0, self.noise_stddev, x.shape)
            x_new = torch.add(x, noise)
        else:
            x_new = x
        return torch.add(torch.mm(x_new, self.w), self.b)

class MyNoisyModule(nn.Module):
    def __init__(self):
        super(MyNoisyModule, self).__init__()
        self.l1 = NoisyLinear(2,4,0.07)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(4,4)
        self.a2 = nn.ReLU()
        self.l3 = nn.Linear(4,1)
        self.a3 = nn.Sigmoid()
        
    def forward(self,x, training=False):
        x = self.l1(x,training)
        x = self.a1(x)
        x = self.l2(x)
        x = self.a2(x)
        x = self.l3(x)
        x = self.a3(x)
        return x

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        pred = self.forward(x)[:,0]
        return (pred>=0.5).float()       

class TwoHiddenLayersWithNoise1(Base, nn.Module):
    def __init__(self, data, x_valid, y_valid, n_train, n_batch, nin=2, nout=1, lr = 0.001):
        super(TwoHiddenLayersWithNoise1, self).__init__(
            data, x_valid, y_valid, n_train, n_batch, nin, nout, lr
        )
        self._model = MyNoisyModule()
        self._optimizer = torch.optim.SGD(self.model.parameters(), lr = self._lr)


data, x_valid, y_valid, n_train, n_batch = create_data()

num_epochs = 200


zero1 = NoHidden1(data, x_valid, y_valid, n_train, n_batch)
history = train(zero1, num_epochs)
visualize_results(history, 'nohidden')

twohidden = TwoHiddenLayers1(data, x_valid, y_valid, n_train, n_batch, 2, 1, 0.015)
history = train(twohidden, num_epochs)
visualize_results(history, 'twohidden')

twohidden_noise = TwoHiddenLayersWithNoise1(data, x_valid, y_valid, n_train, n_batch, 2,1,0.015)
history = train(twohidden_noise, num_epochs)
visualize_results(history, 'twohidden_noise')
