import pathlib
import torch
from torch import nn
from data import load_data
import neptune.new as neptune
from torchvision.models import resnet18
from train import train_model
from helpers import save_architecture_txt


# Initialing the neptune logger.
run = neptune.init(project='sathwik-panchangam/pytorch-deep-learning',
                    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3M2NjMWM4NC02OWMyLTQzMmQtYmYxMC01MmM1NjAyMGRhMjIifQ==",)

# Creating a dictionary for the parameters.
parameters = { 'num_epochs': 3,
                'num_classes': 16,
                'batch_size': 64, 
                'model_name':'Resnet18',
                'lr': 1e-2,
                'weight_decay':1e-5,
                'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}

# Logging hyperparameters.
run['config/hyperparameters'] = parameters

# Set the dataset path
data_dir = '/home/sathwikpanchngam/rnd/Datasets/real_world/asus_combined/'

# Load the dataloaders
dataloaders,class_names = load_data(data_dir=data_dir,batch_size=8,logger=run)

trainloader = dataloaders['train']
testloader = dataloaders['test']
validloader = dataloaders['validation']

# Creating a model for the network
model = resnet18(True)

# Creating a loss function
loss_function = nn.CrossEntropyLoss()

# Creating an optimizer
optimizer = torch.optim.Adam(model.parameters(),lr=parameters['lr'],weight_decay = parameters['weight_decay'])

# Define scheduler
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=4,verbose=True)

# Log the model architecture, loss_function and optimizer
run['config/model'] = type(model).__name__
run['config/criterion'] = type(loss_function).__name__
run['config/optimizer'] = type(optimizer).__name__

# Save model architecture as a text file
save_path = '/home/sathwikpanchngam/rnd/github_projects/SathwikPanchangamRnD/Deeplearning_pipeline/results'
save_architecture_txt(model=model,dir_path=save_path,filename=parameters['model_name'])

# Training results
training_results = train_model(model=model,
                                num_epochs=parameters['num_epochs'],
                                criterion=loss_function,
                                optimizer=optimizer,
                                scheduler=None,
                                dataloader=dataloaders['train'],
                                class_names = class_names ,
                                logger=run)

labels,targets,train_losses,valid_losses = training_results

# Save training results as csv files. Save the losses train, validation etc.

