import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

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
 
        transform_train = transform.Compose([
                transforms.RandomCrop([178,178]),
                transforms.RandomHorizontalFlip(),
                transforms.Resize([64,64]),
                transforms.ToTensor()
            ])
        
        transform = transforms.Compose([
                transforms.CenterCrop([178,178]),
                transforms.Resize([64,64]),
                transforms.ToTensor()
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

        self.batch_size = 32
        torch.manual_seed(1)
        
        self.train_dl = DataLoader(self.sub_train, self.batch_size, shuffle=True)        
        self.valid_dl = DataLoader(self.sub_valid, self.batch_size, shuffle=False)
        self.test_dl = DataLoader(self.celeba_test_dataset, batch_size, shuffle=False)
        

mytrain = CeleSmile()

Hotest = 1
