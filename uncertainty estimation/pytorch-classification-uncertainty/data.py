import torch
from torchvision.datasets.mnist import MNIST

from torchvision.datasets import VisionDataset
import os
from PIL import Image
from numpy import random



import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader , random_split

data_dir = '/home/sathwikpanchngam/rnd/Datasets/real_world/asus_combined/'

transform = transforms.Compose([
        transforms.CenterCrop(10),
        transforms.Resize((32,32)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

data_set = ImageFolder(data_dir,transform=transform)

num_valid = int(np.floor(0.2*len(data_set)))
num_test = int(np.floor(0.1*len(data_set)))
num_train = len(data_set)-num_valid - num_test

train_set,validation_set,test_set = random_split(data_set, [num_train,num_valid,num_test])

train_dataloader = DataLoader(dataset=train_set,batch_size=4,shuffle=True)
validation_dataloader = DataLoader(dataset=validation_set,batch_size=4,shuffle=True)
test_dataloader = DataLoader(dataset=test_set,batch_size=4,shuffle=True)

dataloaders = {
    "train": train_dataloader,
    "val": validation_dataloader,
    "test": test_dataloader
}


'''
class MNIST(VisionDataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    resources = [
        ("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]

    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(MNIST, self).__init__(root, transform=transform,
                                    target_transform=target_transform)
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        #######################################Modification from
        #https://github.com/pytorch/vision/issues/9#issuecomment-304224800
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms

        if self.transform is not None:
            img = self.transform(img)

        random.seed(seed) # apply this seed to target tranfsorms
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.test_file)))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # download files
        for url, md5 in self.resources:
            filename = url.rpartition('/')[2]
            download_and_extract_archive(url, download_root=self.raw_folder, filename=filename, md5=md5)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")

class LabelTransform(object):
    def __init__(self, scale=None):
        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

    @staticmethod
    def get_params(self, scale_ranges):
        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

    def __call__(self, target):
        ret = self.get_params(self, self.scale)
        target = {'label': target, 'scale': ret}
        return target, ret

    def __repr__(self):
        return self.__class__.__name__ + '(scale={0})'.format(self.scale)
 '''       

# class MNIST1(MNIST):
    
#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index

#         Returns:
#             tuple: (image, target,mean_pixel) where target is index of the target class.
#         """
#         img, target = self.data[index], int(self.targets[index])

#         # doing this so that it is consistent with all other datasets
#         # to return a PIL Image
        
#         img = Image.fromarray(img.numpy(), mode='L')
#        # the exta item to be returned
#         #mean_pixel = PIL.ImageStat.Stat(img).mean
#         mean_pixel = 1
#         #######################################Modification from
#         #https://github.com/pytorch/vision/issues/9#issuecomment-304224800
#         seed = np.random.randint(2147483647) # make a seed with numpy generator 
#         random.seed(seed) # apply this seed to img tranfsorms

#         if self.transform is not None:
#             img = self.transform(img)

#         random.seed(seed) # apply this seed to target tranfsorms
#         if self.target_transform is not None:
#             target = self.target_transform(target)
        
#         #sample ={"image":img,"target":target,"mean_pixel",mean_pixel}
#         return img, (target, mean_pixel)



# data_train = MNIST1("./data/mnist",
#                    download=True,
#                    train=True,
#                    transform=transforms.Compose([
#                        transforms.RandomAffine(degrees=0,translate=(0.2,0.2),scale=(0.6,1.0)),
#                        transforms.Resize((28, 28)),
#                        transforms.ToTensor()]))
#                     #target_transform=LabelTransform(scale=(0.6,1.0)))
#                     #target_transform=transforms.Compose([
#                     #   LabelTransform(scale=(0.6,1.0)),
#                     #   transforms.ToTensor()]))
#                       # transforms.Normalize((0.5,), (1.0,))]))

# data_val = MNIST1("./data/mnist",
#                  train=False,
#                  download=True,
#                  transform=transforms.Compose([
#                      #transforms.RandomAffine(degrees=0,translate=(0.2,0.2),scale=(0.5,1.0)),
#                      transforms.Resize((28, 28)),
#                      transforms.ToTensor()]))
#                  #target_transform=LabelTransform())
#                  #target_transform=transforms.Compose([
#                  #  LabelTransform()]))
#                      #transforms.Normalize((0.5,), (1.0,))]))

# dataloader_train = DataLoader(
#     data_train, batch_size=1000, shuffle=True, num_workers=8)
# dataloader_val = DataLoader(data_val, batch_size=1000, num_workers=8)

# dataloaders = {
#     "train": dataloader_train,
#     "val": dataloader_val,
# }

# digit_zero, _ = data_val[3]
# digit_one, _ = data_val[2]
# digit_two, _ = data_val[1]
# digit_three, _ = data_val[18]
# digit_four, _ = data_val[6]
# digit_five, _ = data_val[8]
# digit_six, _ = data_val[21]
# digit_seven, _ = data_val[0]
# digit_eight, _ = data_val[110]
# digit_nine, _ = data_val[7]



