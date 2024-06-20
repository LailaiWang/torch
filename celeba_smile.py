import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
class CeleSmile(object):
    def __init__(self):
        super(CeleSmile, self).__init__()
        self.prepare_data()

    def prepare_data(self):
        import os
        target = os.path.join(os.getcwd(), 'celeba')
        if os.path.exists(target):
            to_download = False
        else:
            to_download = True
        
        image_path = './'
 
        transform_train = transforms.Compose([
                transforms.RandomCrop([178,178]),
                transforms.RandomHorizontalFlip(),
                transforms.Resize([64,64]),
                transforms.ToTensor()
            ])
        
        transform = transforms.Compose([
                transforms.CenterCrop([178,178]),
                transforms.Resize([64,64]),
                transforms.ToTensor() # torch work with tensor data set
            ])

        get_smile = lambda attr: attr[31]
               
        self.celeba_train_dataset = torchvision.datasets.CelebA(
                image_path, split='train',
                target_type = 'attr', download=to_download,
                transform = transform_train, target_transform = get_smile
            )

        self.celeba_valid_dataset = torchvision.datasets.CelebA(
                image_path, split='valid',
                target_type = 'attr', download=to_download,
                transform = transform, target_transform = get_smile
            )

        self.celeba_test_dataset = torchvision.datasets.CelebA(
                image_path, split='test',
                target_type = 'attr', download=to_download,
                transform=transform, target_transform = get_smile
            )

        # now get a subset of the data
        # we will use the whole data after we figure out how to train with multiple gpus
        
        from torch.utils.data import Subset

        self.sub_train = Subset(self.celeba_train_dataset, torch.arange(16000))
        self.sub_valid = Subset(self.celeba_valid_dataset, torch.arange(1000))

        self.batch_size = 2048
        torch.manual_seed(1)
        
        self.train_dl = DataLoader(self.sub_train, self.batch_size, shuffle=True)        
        self.valid_dl = DataLoader(self.sub_valid, self.batch_size, shuffle=False)
        self.test_dl = DataLoader(self.celeba_test_dataset, self.batch_size, shuffle=False)
    
    def prepare_model(self):
        '''
        Random dropout can be considered as a regularization method to
        prevent overfitting
        '''
        model = nn.Sequential()
        model.add_module('conv1',
                nn.Conv2d(in_channels=3, out_channels = 32,
                    kernel_size = 3, padding = 1
                )
            )
        model.add_module('relu1', nn.ReLU())
        model.add_module('pool1', nn.MaxPool2d(kernel_size=2))
        model.add_module('dropout1', nn.Dropout(p=0.5))

        model.add_module('conv2',
                nn.Conv2d(in_channels=32, out_channels = 64,
                    kernel_size = 3, padding = 1
                )
            )
        model.add_module('relu2', nn.ReLU())
        model.add_module('pool2', nn.MaxPool2d(kernel_size=2))
        model.add_module('dropout2', nn.Dropout(p=0.5))

        model.add_module('conv3',
                nn.Conv2d(in_channels=64, out_channels = 128,
                    kernel_size = 3, padding = 1
                )
            )
        model.add_module('relu3', nn.ReLU())
        model.add_module('pool3', nn.MaxPool2d(kernel_size=2))
        
        model.add_module('conv4',
                nn.Conv2d(in_channels=128, out_channels = 256,
                    kernel_size = 3, padding = 1
                )
            )
        model.add_module('relu4', nn.ReLU())
        model.add_module('pool4', nn.AvgPool2d(kernel_size=8))
        model.add_module('flatten', nn.Flatten())
        
        model.add_module('fc', nn.Linear(256,1))
        model.add_module('sigmoid', nn.Sigmoid())
        
        # move to device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('device is ', device)
        self._model = model.to(device)
        self._loss = nn.BCELoss().to(device)

        self._optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    @property
    def loss(self):
        return self._loss

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def model(self):
        return self._model         
    
    def train(self, num_epochs):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        loss_hist_train = [0] * num_epochs
        accuracy_hist_train = [0] * num_epochs
        loss_hist_valid = [0] * num_epochs
        accuracy_hist_valid = [0] * num_epochs
        for epoch in range(num_epochs):
            self.model.train()
            for x_batch, y_batch in self.train_dl:
                
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                pred = self.model(x_batch)[:,0]
                loss = self.loss(pred, y_batch.float())
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss_hist_train[epoch] += loss.item() * y_batch.size(0)
                is_correct = ((pred>=0.5).float() == y_batch).float()
                accuracy_hist_train[epoch] += is_correct.sum()
            loss_hist_train[epoch] /= len(self.train_dl.dataset)
            accuracy_hist_train[epoch] /= len(self.train_dl.dataset)
        
            self.model.eval()

            with torch.no_grad():
                for x_batch, y_batch in self.valid_dl:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    pred = self.model(x_batch)[:,0]
                    loss = self.loss(pred, y_batch.float())
                    loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
                    is_correct = ((pred>=0.5).float() == y_batch).float()
                    accuracy_hist_valid[epoch] += is_correct.sum()
            loss_hist_valid[epoch] /= len(self.valid_dl.dataset)
            accuracy_hist_valid[epoch] /= len(self.valid_dl.dataset) 
            print(f'Epoch {epoch+1} accuracy: {accuracy_hist_train[epoch]:.4f} va_acc {accuracy_hist_valid[epoch]:.4f}' )          
        return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid

    def visualize_hist(self, hist):
        import matplotlib.pyplot as plt
        x_arr = np.array(len(hist[0])) + 1
        fig = plt.figure(figsize(12,4))
        ax = fig.add_subplot(1,2,1)
        ax.plot(x_arr, hist[0],'-o', label='Train loss')
        ax.plot(x_arr, hist[1],'--<', label='Validation loss')
        ax.legend(fontsize=15)
        ax = fig.add_subplot(1,2,2)
        ax.plot(x_arr, hist[2],'-o', label='Train Acc')
        ax.plot(x_arr, hist[3],'--<', label='Validation Acc')
        ax.legend(fontsize=15)
        ax.set_xlabel('Epoch', size=15)
        ax.set_ylabel('Acc', size=15)
        plt.show()
        plt.savefig('stats.png')

if __name__==  '__main__':            
    mytrain = CeleSmile()
    mytrain.prepare_data()
    mytrain.prepare_model()
    hist = mytrain.train(30)
    mytrain.visualize_hist(hist)
