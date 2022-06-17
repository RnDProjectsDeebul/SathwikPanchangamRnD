import pathlib
import torch
from torch import nn
from data import load_data
from train import Training
from metrics_helper import Metrics
import neptune.new as neptune

from torchvision.models import resnet18

# Initialing the neptune logger.
run = neptune.init(project='sathwik-panchangam/pytorch-deep-learning',api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3M2NjMWM4NC02OWMyLTQzMmQtYmYxMC01MmM1NjAyMGRhMjIifQ==",)

# Creating a dictionary for the parameters.
parameters = { 'lr': 1e-2,
               'batch_size': 64,
               'num_classes': 16,
               'num_epochs': 3,
               'model_name':'Resnet18',
               'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}

# Logging hyperparameters.
run['config/hyperparameters'] = parameters

# Loading the data
data_dir = '/home/sathwikpanchngam/rnd/Datasets/real_world/asus_combined'

dataloaders = load_data(data_dir=data_dir,logger=run)

train_loader = dataloaders['train']
test_loader = dataloaders['test']
valid_loader = dataloaders['validation']


# Creating a model for the network
model = resnet18(pretrained=True)

# Creating a loss function
loss_function = nn.CrossEntropyLoss()

# Creating an optimizer
optimizer = torch.optim.Adam(model.parameters(),parameters['lr'])

# Logging the model architecture, lossfunction and optimizer.
run['config/model'] = type(model).__name__
run['config/criterion'] = type(loss_function).__name__
run['config/optimizer'] = type(optimizer).__name__

architecture = parameters['model_name']
with open(f"./{architecture}_arch.txt", "w") as f:
    f.write(str(model))

# Check for the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


train_model = Training(num_epochs=parameters['num_epochs'],model=model,criterion=loss_function,optimizer=optimizer,trainloader=train_loader,testloader=test_loader,logger=run)

training_results = train_model.train(device=device)


labels = training_results[2]
targets = training_results[3]
train_loss = training_results[0]
test_loss = training_results[1]

root = pathlib.Path(data_dir)
classes = sorted([j.name.split('/')[-1] for j in root.iterdir()])
print(classes)
# METRICS
metrics = Metrics(targets,labels,classes,train_loss,test_loss)

# Accuracy
accuracy = metrics.get_accuracy()
print("Accuracy : ",accuracy)

# Precision
precision = metrics.get_precision_score()
print("Precision : ",precision)

# Confusion Matrix
conf_mat1 = metrics.plot_confusion_matrix1()
conf_mat2 = metrics.plot_confusion_matrix2()

# Classification Report.
class_report = metrics.get_classification_report()
print(class_report)

# F1 score
f1_score = metrics.get_f1_score()
print("F1 Score : ",f1_score)

# Recall
recall = metrics.get_recall_score()
print("Recall : ",recall)

# Plot losses
train_loss = metrics.plot_train_loss()
test_loss = metrics.plot_test_loss()

run.stop()