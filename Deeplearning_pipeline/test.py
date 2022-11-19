import torch
from torchvision.models import resnet18,mobilenet_v2
from torch import nn
from data import load_test_data,load_data
from helpers import get_model, test_one_epoch, get_brier_score, get_expected_calibration_error,get_best_worst_predictions,plot_predicted_images
from helpers import get_accuracy_score,get_precision_score,get_recall_score,get_f1_score,get_classification_report,plot_confusion_matrix1
import torchvision
import copy
import numpy as np
import sys
import neptune.new as neptune
import matplotlib.pyplot as plt


# Set the dataset path
data_dir = '/home/sathwikpanchngam/rnd/Final_experiments/datasets/real_world/asus_combined'

# Creating a dictionary for the parameters. Model names : Resnet18, Mobilenetv2, Squeezenet
parameters = {  'num_classes': 15,
                'batch_size': 64, 
                'model_name':'Resnet18',
                'loss_function': 'Crossentropy',
                'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}

# Set the results path
save_path = '/home/sathwikpanchngam/rnd/Final_experiments/results/real_world/robocup/cross_entropy/resnet/'

# Set the model path
model_path = str(save_path)+'model.pth'

# Set dataset type for the file name to save the best and worst preds plot 
data_set_type = 'real_world'

best_preds_name = 'best_pred'+str(parameters['model_name'])+ str(parameters['loss_function'])+ data_set_type
worst_preds_name = 'worst_pred'+str(parameters['model_name'])+ str(parameters['loss_function'])+ data_set_type

# Set name for confusion matrix plot
confusion_matrix_name = 'confusion_matrix'+str(parameters['model_name'])+ str(parameters['loss_function'])+ data_set_type

# Initialing the neptune logger.
logger = True  # spanch2s
if logger:
    # run = neptune.init('Provide your neptune ai key')
    run = neptune.init(
    project="sathwik-panchangam/RnD-Experiment-Results",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3M2NjMWM4NC02OWMyLTQzMmQtYmYxMC01MmM1NjAyMGRhMjIifQ==",
    tags = ["realworld","cross_entropy","Resnet18","Robocup"],
    name= "testing" + "-" + str(data_set_type) + "-" + str(parameters['model_name']) + "-" + str(parameters['loss_function']),
    )
else:
    run = None

# set the device
device = parameters['device']

# Get the model/create the dnn model
model = get_model(parameters['model_name'],num_classes=parameters['num_classes'])

# Load the best saved model from the training results.
model.load_state_dict(torch.load(model_path))
model.eval()
model.to(device=device)

# Load the dataloaders for real world
dataloader,class_names = load_data(data_dir=data_dir,batch_size=parameters['batch_size'],logger=run)

# Load the dataloader for synthetic data.
# dataloader,class_names = load_test_data(data_dir=data_dir,batch_size=parameters['batch_size'],logger=run)


test_loader = dataloader['test']
print("Number of test images : ",len(test_loader)*parameters['batch_size'])


results = test_one_epoch(model=model,dataloader=test_loader,device=device,loss_function=parameters['loss_function'])

true_labels = results[0]
pred_labels = results[1]
confidences = results[2]
probabilities_vector = results[3]
images = results[4]
labels = results[5]

# Get best and worst predictions
best_predictions,worst_predictions = get_best_worst_predictions(confidences)

# classification metrics
accuracy_score = round(get_accuracy_score(true_labels=true_labels,predicted_labels=pred_labels),3)
precision_score = round(get_precision_score(true_labels,pred_labels),3)
recall_score = round(get_recall_score(true_labels,pred_labels),3)
f1_score = round(get_f1_score(true_labels,pred_labels),3)
classification_report = get_classification_report(true_labels,pred_labels,class_names)

# Uncertainty metrics
brier_score = get_brier_score(y_true=true_labels,y_pred_probs=probabilities_vector)
expected_calibration_error = get_expected_calibration_error(y_true=true_labels,y_pred=probabilities_vector)

# Plot confusion matrix
confusion_mat_fig = plot_confusion_matrix1(true_labels=true_labels,
                                            predicted_labels=pred_labels,
                                            class_names=class_names,
                                            results_path=save_path,
                                            plot_name=confusion_matrix_name)

# Plot the best and worst predictions.
best_fig = plot_predicted_images(predictions=pred_labels,
                                 confidences_pred_images=confidences,
                                 images=images,
                                 labels=labels,
                                 class_names=class_names,
                                 plot_preds=best_predictions,
                                 results_path=save_path,
                                 plot_name= best_preds_name
                                 )

worst_fig = plot_predicted_images(predictions=pred_labels,  
                                  confidences_pred_images=confidences,
                                  images=images,
                                  labels=labels,
                                  class_names=class_names,
                                  plot_preds=worst_predictions,
                                  results_path=save_path,
                                  plot_name= worst_preds_name
                                  )


# Print results
print("Accuracy score : ", accuracy_score)
print("Precision score : ", precision_score)
print("Recall score : ", recall_score)
print("F1 score : ", f1_score)
print("Brier Score : ", brier_score)
print("Expected calibration error : ", expected_calibration_error)
print('--'*20)
print("\nClassification report : ",classification_report)

if run !=None:
    # Logging the results.

    # log model parameters
    run['config/hyperparameters'] = parameters
    run['config/model'] = type(model).__name__
    
    # log metrics
    run['metrics/accuracy'] = accuracy_score
    run['metrics/precision_score'] = precision_score
    run['metrics/recall_score'] = recall_score
    run['metrics/f1_score'] = f1_score
    run['metrics/brier_score'] = brier_score
    run['metrics/expected_calibration_error'] = expected_calibration_error
    run['metrics/classification_report'] = classification_report

    # log images
    run['metrics/images/confusion_matrix'].upload(confusion_mat_fig)
    run['metrics/images/best_predictions'].upload(best_fig)
    run['metrics/images/worst_predictions'].upload(worst_fig)


# Dropout at test time.

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def get_monte_carlo_predictions(data_loader,
                                forward_passes,
                                model,
                                n_classes,
                                n_samples):

    dropout_predictions = np.empty((0, n_samples, n_classes))
    softmax = nn.Softmax(dim=1)
    for i in range(forward_passes):
        predictions = np.empty((0, n_classes))
        model.eval()
        enable_dropout(model)
        for i, (image, label) in enumerate(data_loader):

            image = image.to(torch.device('cuda'))
            with torch.no_grad():
                output = model(image)
                output = softmax(output) # shape (n_samples, n_classes)
            predictions = np.vstack((predictions, output.cpu().numpy()))

        dropout_predictions = np.vstack((dropout_predictions,
                                         predictions[np.newaxis, :, :]))
        # dropout predictions - shape (forward_passes, n_samples, n_classes)
    
    # Calculating mean across multiple MCD forward passes 
    mean = np.mean(dropout_predictions, axis=0) # shape (n_samples, n_classes)

    # Calculating variance across multiple MCD forward passes 
    variance = np.var(dropout_predictions, axis=0) # shape (n_samples, n_classes)

    epsilon = sys.float_info.min
    # Calculating entropy across multiple MCD forward passes 
    entropy = -np.sum(mean*np.log(mean + epsilon), axis=-1) # shape (n_samples,)

    # Calculating mutual information across multiple MCD forward passes 
    mutual_info = entropy - np.mean(np.sum(-dropout_predictions*np.log(dropout_predictions + epsilon),
                                            axis=-1), axis=0) # shape (n_samples,)









# train_data = copy.deepcopy(train_loader)
# dataiter = iter(test_data)
# images, labels = dataiter.next()

# # print images
# imshow(torchvision.utils.make_grid(images))
# print('GroundTruth: ', ' '.join('%s' % class_names[labels[j]] for j in range(4)))


# outputs = model(images)
# _, predicted = torch.max(outputs, 1)
# print('Predicted: ', ' '.join('%s' % class_names[predicted[j]] for j in range(4)))




# confidance = []
# classes_list = []

# correct = 0
# total = 0
# with torch.no_grad():
#     for data in test_loader:
#         images, labels = data
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
        
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#         conf, classes = torch.max(outputs, 1) # _ in above line and conf are same
#         # conf has 4 batches with 64 images each
#         confidance.append(conf)
#         classes_list.append(classes)

#         # print(conf)
# print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
# print('\n Classes list : ', classes_list[0])
# print('\n confidance : ', confidance[0])
# print("\n total",torch.sum(confidance[0]))