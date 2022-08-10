import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader , random_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


# data_dir = '/home/sathwikpanchngam/rnd/Datasets/real_world/asus_combined/'

def load_data(data_dir,batch_size,logger):

    transform = transforms.Compose([
        transforms.Resize((96,96)),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],
        [0.5,0.5,0.5])
        ])

    random_seed = 42
    torch.manual_seed(random_seed)

    data_set = ImageFolder(data_dir,transform=transform)
    num_valid = int(np.floor(0.2*len(data_set)))
    num_test = int(np.floor(0.1*len(data_set)))
    num_train = len(data_set)-num_valid - num_test

    train_set,validation_set,test_set = random_split(data_set, [num_train,num_valid,num_test])

    train_dataloader = DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True,num_workers=8)
    validation_dataloader = DataLoader(dataset=validation_set,batch_size=batch_size,shuffle=True,num_workers=8)
    test_dataloader = DataLoader(dataset=test_set,batch_size=batch_size,shuffle=True,num_workers=8)

    dataloaders = {
        "train": train_dataloader,
        "val": validation_dataloader,
        "test": test_dataloader }

    class_names = data_set.classes

    dataset_parameters={'transforms': transform,
                        'random_seed':random_seed,
                        'train_size': len(train_dataloader)*batch_size,
                        'val_size': len(validation_dataloader)*batch_size,
                        'test_size':len(test_dataloader)*batch_size}

    # logger['config/dataset/'] = dataset_parameters

    return dataloaders,class_names

def imshow(img):
    img = img/2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()




# if __name__ == '__main__':
#     batch_size = 8
#     dataloaders,class_names = load_data(data_dir=data_dir)
#     print(class_names)
#     dataiter = iter(dataloaders['train'])
#     images,labels = dataiter.next()
#     imshow(make_grid(images))
#     print(' '.join(f'{class_names[labels[j]]:5s}'for j in range(batch_size)))

