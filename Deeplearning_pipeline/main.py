import pathlib
import torch
from torch import nn
from data import load_data
# from data1 import load_data1
import neptune.new as neptune
from torchvision.models import resnet18
from train import train_model
from helpers import save_architecture_txt, one_hot_embedding


from losses import mse_loss,edl_loss,edl_mse_loss

# Initialing the neptune logger.
run = neptune.init('Provide your neptune ai key')

# Set the dataset path for train data and saving the results


data_dir = 'set_path/training_dataset'
save_path = 'set_path/to_save_results'

# Creating a dictionary for the parameters.
parameters = { 'num_epochs': 40,
                'num_classes': 11,
                'batch_size': 32, 
                'model_name':'Resnet18',
                'lr': 1e-3,
                'weight_decay':1e-5,
                'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}

# Logging hyperparameters.
run['config/hyperparameters'] = parameters


# Load the dataloaders
dataloader,class_names = load_data(data_dir=data_dir,batch_size=parameters['batch_size'],logger=run)

# Creating a model for the network
model = resnet18(weights = 'DEFAULT')
model.fc = nn.Linear(in_features=512,out_features=parameters['num_classes'])


uncertainty = True
# Creating a loss function
if uncertainty:
    loss_function = edl_mse_loss
else:
    loss_function = nn.CrossEntropyLoss()


# Creating an optimizer
optimizer = torch.optim.Adam(model.parameters(),lr=parameters['lr'],weight_decay = parameters['weight_decay'])

# Define scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,T_0=10,T_mult=2,eta_min=1e-4)



# Log the model architecture, loss_function and optimizer
run['config/model'] = type(model).__name__
run['config/criterion'] = type(loss_function).__name__
run['config/optimizer'] = type(optimizer).__name__

# Save model architecture as a text file

save_architecture_txt(model=model,dir_path=save_path,filename=parameters['model_name'])

# Training results
training_results = train_model(model=model,
                                num_epochs=parameters['num_epochs'],
                                uncertainty = uncertainty,
                                criterion=loss_function,
                                optimizer=optimizer,
                                scheduler=lr_scheduler,
                                dataloaders=dataloader,
                                class_names = class_names ,
                                logger=run)

best_model,train_losses,valid_losses = training_results

