from torch.utils.data import DataLoader
from custom_dataset import CustomDataset
from torchvision import transforms


def load_data(data_dir,logger):
    # Create transform for preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32,32)),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        transforms.RandomHorizontalFlip()
        ])

    # from dataloader import trainloader, testloader

    train_set = CustomDataset(root=data_dir,train=True,transform=transform)
    trainloader = DataLoader(train_set, batch_size=64, shuffle=True)


    test_set = CustomDataset(root=data_dir, test=True, transform=transform)
    testloader = DataLoader(test_set, batch_size=64, shuffle=True)

    valid_set = CustomDataset(root=data_dir,validation=True,transform=transform)
    validloader = DataLoader(valid_set,batch_size=64,shuffle=True)

    dataset_size = {'train':len(train_set), 'test':len(test_set),'valid':len(valid_set)}

    logger['config/dataset/transforms'] = transform
    logger['config/dataset/size'] = dataset_size


    dataloaders = {"train":trainloader,"test":testloader,"validation":validloader}

    return dataloaders