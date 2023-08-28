import torch
from torch import nn
from data import load_data, load_synthetic_data
# from data1 import load_data1
import neptune as neptune
from torchvision.models import resnet18, mobilenet_v2,vgg16,vgg19,mobilenet_v3_small
from train import train_model
from helpers import save_architecture_txt,get_model
from losses import mse_loss,edl_loss,edl_mse_loss
import os
import timm
import torchvision
from torchvision import transforms

print("Imported everything sucessfully")

# Set the dataset path for train data and saving the results
data_dir = '/path/for/dataset/'
# Save path for results
save_path = '/path/for/saving/results/'

# names for other files
data_set_type = 'normal'

# Creating a dictionary for the parameters.
# Creating a dictionary for the parameters. 
# Model names : Resnet18, Mobilenetv2, Mobilenetv3_small,Resnet34
# Loss funcitons: Crossentropy, Evidential
parameters = {'num_epochs': 30,
              'num_classes': 35,
              'batch_size': 512,
              'model_name': 'Resnet18',
              'loss_function': 'Evidential',
              'Dropout': False,
              'lr': 0.001,
              'weight_decay': 1e-3,
              'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
              'logger':False
              }

# Define scheduler
# lr_scheduler = None
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

if parameters['Dropout']==True:
    # Dropout conditon name
    condition_name = str(data_set_type)+'_'+str(parameters['loss_function'])+'_'+str('Dropout')
    # Get the model/create the dnn model
    model = timm.create_model('resnet18',pretrained=False,num_classes=parameters['num_classes'],drop_rate=0.5)
    
else:
    condition_name = str(data_set_type)+'_'+str(parameters['loss_function'])+'_'+str(parameters['model_name'])
    # Get the model/create the dnn model
    model = get_model(parameters['model_name'],num_classes=parameters['num_classes'],weights=True)

uncertainty = False
# Create the loss function
if parameters['loss_function'] == 'Crossentropy':
    loss_function = nn.CrossEntropyLoss()
elif parameters['loss_function'] == 'Evidential':
    loss_function = edl_mse_loss
    # uncertainty required or not True if evidential loss
    uncertainty = True
else:
    raise NotImplementedError

# Creating an optimizer
optimizer = torch.optim.Adam(model.parameters(),lr=parameters['lr'],weight_decay = parameters['weight_decay'])

# Initialing the neptune logger.
if parameters['logger']:
    # run = neptune.init('Provide your neptune ai key')
    run = neptune.init(
    project="provide the name of the project in neptune ai",
    api_token='provide your api token',
    tags = [str(data_set_type),str(parameters['loss_function']),str(parameters['model_name']),"Robocup","Training"],
    name= "Training" + "-" + str(data_set_type) + "-" + str(parameters['model_name']) + "-" + str(parameters['loss_function']),
    )
    # Logging hyperparameters.
    run['config/hyperparameters'] = parameters
    # Log the model archit  ecture, loss_function and optimizer
    run['config/model'] = type(model).__name__
    run['config/criterion'] = parameters['loss_function']
    run['config/optimizer'] = type(optimizer).__name__
else:
    run = None

# Load the dataloaders for real world
# dataloader,class_names = load_data(data_dir=data_dir,batch_size=parameters['batch_size'],logger=run)

# Load the dataloader for synthetic data.
dataloader,class_names = load_synthetic_data(data_dir=data_dir,batch_size=parameters['batch_size'],logger=run)

batch_size = parameters['batch_size']
trainloader = dataloader['train']
testloader = dataloader['val']
# class_names = trainset.classes
print("Class names : ",class_names)
print("No of classes : ", len(class_names))
print("No of train images : ", len(trainloader)*batch_size)
print("No of test images : ", len(testloader)*batch_size)

#######################################################

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
                                results_file_path =save_path,
                                condition_name=condition_name)

best_model = training_results

torch.save(best_model.state_dict(),save_path+str(condition_name)+str('_model.pth'))

# TODO: ADD early stopping in training 
