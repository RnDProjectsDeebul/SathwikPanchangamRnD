import torch
from torch import nn
from data import load_data, load_synthetic_data
# from data1 import load_data1
import neptune.new as neptune
from torchvision.models import resnet18, mobilenet_v2
from train import train_model
from helpers import save_architecture_txt,get_model
from losses import mse_loss,edl_loss,edl_mse_loss


# Set the dataset path for train data and saving the results

# Real world datasets
data_dir = '/home/sathwikpanchngam/rnd/Final_experiments/datasets/real_world/asus_combined'

# Synthetic datasets 
# data_dir = '/home/sathwikpanchngam/rnd/Mid_term/Datasets/asus_different_lighting/normal'

# Save path for results
# save_path = '/home/sathwikpanchngam/rnd/Final_experiments/results/real_world/robocup/cross_entropy/mobilenet/'
save_path = '/home/sathwikpanchngam/rnd/Final_experiments/results/real_world/robocup/cross_entropy/mobilenet/'

# data_dir = 'set_path/training_dataset'
# save_path = 'set_path/to_save_results'

# Creating a dictionary for the parameters.
parameters = { 'num_epochs': 20,
                'num_classes': 15,
                'batch_size': 64, 
                'model_name':'Mobilenetv2',
                'loss_function':'Crossentropy',
                'lr': 1e-3,
                'weight_decay':1e-5,
                'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}

# 'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get the model/create the dnn model
model = get_model(parameters['model_name'],num_classes=parameters['num_classes'])

# Create the loss function
if parameters['loss_function'] == 'Crossentropy':
    loss_function = nn.CrossEntropyLoss()
elif parameters['loss_function'] == 'Evidential':
    loss_function = edl_mse_loss
else:
    raise NotImplementedError


# uncertainty required or not True if evidential loss
uncertainty = False

# Creating an optimizer
optimizer = torch.optim.Adam(model.parameters(),lr=parameters['lr'],weight_decay = parameters['weight_decay'])


# Define scheduler
# lr_scheduler = None
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,T_0=10,T_mult=2,eta_min=1e-4)

# Initialing the neptune logger.
logger = True  # spanch2s
if logger:
    # run = neptune.init('Provide your neptune ai key')
    run = neptune.init(
    project="sathwik-panchangam/Pytorch-classification",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3M2NjMWM4NC02OWMyLTQzMmQtYmYxMC01MmM1NjAyMGRhMjIifQ==",
    )
    # Logging hyperparameters.
    run['config/hyperparameters'] = parameters
    # Log the model architecture, loss_function and optimizer
    run['config/model'] = type(model).__name__
    run['config/criterion'] = parameters['loss_function']
    run['config/optimizer'] = type(optimizer).__name__
else:
    run = None

# Load the dataloaders for real world
dataloader,class_names = load_data(data_dir=data_dir,batch_size=parameters['batch_size'],logger=run)

# Load the dataloader for synthetic data.
# dataloader,class_names = load_synthetic_data(data_dir=data_dir,batch_size=parameters['batch_size'],logger=run)


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
                                logger=run,
                                results_file_path =save_path)

best_model = training_results

torch.save(best_model.state_dict(),save_path+str('/model.pth'))


# TODO: Add conusion matrix plots, losses plots using seaborn and pandas and save the results 
# TODO: show_sample_images. 
# TODO: (future work) add arguments from the commandline as a json file