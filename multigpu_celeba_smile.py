import os
import sys

import torch
import torch.nn as nn

import torch.nn.functional as F

import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group, all_reduce, ReduceOp

import numpy as np

def ddp_setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend="nccl", rank=rank, world_size = world_size)
    torch.cuda.set_device(rank)

class CeleSmile(object):
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        valid_data: DataLoader, 
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        save_every: int                
    ) -> None:
        super(CeleSmile, self).__init__()
        self.gpu_id = gpu_id
        self.train_data = train_data
        self.valid_data = valid_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model = DDP(model.to(gpu_id), device_ids=[gpu_id]) # set GPU device
        
        self._set_up_statistics()        

    def _set_up_statistics(self):
        self._loss = torch.zeros([1], device=self.gpu_id)

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.binary_cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch):
        b_sz = self.train_data.batch_size
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz}| Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        self.model.train()
        for source, targets in self.train_data:
            source = source.to(self.gpu_id) # copy to GPU
            targets = targets.reshape(-1,1).float().to(self.gpu_id) # copy to GPU
            self._run_batch(source, targets)

    def _eval_epoch(self, epoch):
        self.valid_data.sampler.set_epoch(epoch)
        self.model.eval()
        
        with torch.no_grad():
            for source, targets in self.valid_data:
                source = source.to(self.gpu_id)
                targets = targets.reshape(-1,1).float().to(self.gpu_id)
                output = self.model(source)
                loss = F.binary_cross_entropy(output, targets)
                self._loss[0] = loss.item() * targets.size(0) # un-average this
        # reduce to 
        all_reduce(self._loss, op=ReduceOp.SUM)
        # total 
        divi = self._loss[0]/len(self.valid_data.dataset)
        print(f'Epoch {epoch} | current loss {divi}')        

    def _save_checkpoint(self, epoch):
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def train(self, max_epochs: int):
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            self._eval_epoch(epoch)
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_checkpoint(epoch)

def load_train_objs():
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
           
    celeba_train_dataset = torchvision.datasets.CelebA(
            image_path, split='train',
            target_type = 'attr', download=to_download,
            transform = transform_train, target_transform = get_smile
        )

    celeba_valid_dataset = torchvision.datasets.CelebA(
            image_path, split='valid',
            target_type = 'attr', download=to_download,
            transform = transform, target_transform = get_smile
        )

    celeba_test_dataset = torchvision.datasets.CelebA(
            image_path, split='test',
            target_type = 'attr', download=to_download,
            transform=transform, target_transform = get_smile
        )

    # now get a subset of the data
    # we will use the whole data after we figure out how to train with multiple gpus
    
    from torch.utils.data import Subset

    sub_train = Subset(celeba_train_dataset, torch.arange(16000))
    sub_valid = Subset(celeba_valid_dataset, torch.arange(1000))

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
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    return sub_train, sub_valid, model, optimizer

def prepare_dataloader(dataset: torchvision.datasets, batch_size: int):
    return DataLoader(
        dataset,
        batch_size = batch_size,
        pin_memory = True, # pinned memory for asychronous memory copy
        shuffle = False,
        sampler = DistributedSampler(dataset)
    )

def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    print(f'done set up ddp on rank {rank} of world_size {world_size}')
    dataset, dataset_v, model, optimizer = load_train_objs()
    print(f'done set up model on rank {rank} of world_size {world_size}')
    train_data = prepare_dataloader(dataset, batch_size)
    valid_data = prepare_dataloader(dataset_v, batch_size)
    trainer = CeleSmile(model, train_data, valid_data, optimizer, rank, save_every)
    trainer.train(total_epochs)
    destroy_process_group()
    
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
