import torch
from torch import nn
from data import load_test_data,load_data,load_synthetic_data
from helpers import get_model, test_one_epoch, get_brier_score, get_expected_calibration_error,get_best_worst_predictions,plot_predicted_images
from helpers import get_accuracy_score,get_precision_score,get_recall_score,get_f1_score,get_classification_report,plot_confusion_matrix1
import os
import numpy as np
import pandas as pd
import neptune as neptune
import matplotlib.pyplot as plt
from helpers import get_multinomial_entropy,get_dirchlet_entropy
import torchvision
from torchvision import transforms

# Set the required paths
data_dir = '/path/test_data/'
models_path = '/path/trained_models/'
save_path = '/path/test_results/'

# Creating a dictionary for the parameters. 
# Model names : Resnet18, Mobilenetv2, Mobilenetv3_small,Resnet34
# Loss funcitons: Crossentropy, Evidential
parameters = {  'num_classes': 35,
                'batch_size': 64, 
                'model_name':'Resnet18',
                'loss_function': 'Evidential',
                'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
                'logger': False,
             }

# Set dataset type for the file name to save the best and worst preds plot 
data_set_type = 'distance' # blur,normal, etc..

# Set the model path
model_path = str(models_path)+'normal_'+parameters['loss_function']+'_Resnet18_model.pth'

condition_name = str(data_set_type)+'_'+str(parameters['loss_function'])+'_'+str(parameters['model_name'])
print(condition_name)

best_preds_name = 'best_pred_'+condition_name
worst_preds_name = 'worst_pred_'+condition_name

# Set name for confusion matrix plot
confusion_matrix_name = 'confusion_matrix_'+str(parameters['model_name'])+'_'+ str(parameters['loss_function'])+ '_'+data_set_type

# Initialing the neptune logger.
if parameters['logger']:
    # run = neptune.init('Provide your neptune ai key')
    run = neptune.init(
    project="provide project name",
    api_token="provide api key",
    tags = [str(data_set_type),str(parameters['loss_function']),str(parameters['model_name']),"Robocup","Testing"],
    name= "testing" + "-" + str(data_set_type) + "-" + str(parameters['model_name']) + "-" + str(parameters['loss_function']),
    )
else:
    run = None

# set the device
device = parameters['device']

# Get the model/create the dnn model
model = get_model(parameters['model_name'],num_classes=parameters['num_classes'],weights=False)

# Load the best saved model from the training results.
model.load_state_dict(torch.load(model_path))
model.eval()
model.to(device=device)

if data_set_type == 'normal':
    # Load the dataloader for synthetic data.
    dataloader,class_names = load_synthetic_data(data_dir=data_dir,batch_size=parameters['batch_size'],logger=run)
else:
    # Load the dataloader for synthetic data.
    dataloader,class_names = load_test_data(data_dir=data_dir,batch_size=parameters['batch_size'],logger=run)


test_loader = dataloader['val']

# dataloader = {"train": train_loader, "val": test_loader}
print("Class names : ",class_names)
print("No of classes : ", len(class_names))
# print("Number of train images : ",len(train_loader)*parameters['batch_size'])

# print(test_loader.sampler)

print("Number of test images : ",len(test_loader)*parameters['batch_size'])



results = test_one_epoch(model=model,
                         dataloader=test_loader,
                         device=device,
                         loss_function=parameters['loss_function']
                         )

# seperate the results
true_labels = results['true_labels']
pred_labels = results['pred_labels']
confidences = results['confidences']
probabilities = results['probabilities']
model_output = results['model_output']
images_list = results['images_list']
image_paths = results['image_paths']

# Get best and worst predictions
best_predictions,worst_predictions = get_best_worst_predictions(confidences)

# classification metrics
accuracy_score = round(get_accuracy_score(true_labels=true_labels,predicted_labels=pred_labels),3)
precision_score = round(get_precision_score(true_labels,pred_labels),3)
recall_score = round(get_recall_score(true_labels,pred_labels),3)
f1_score = round(get_f1_score(true_labels,pred_labels),3)
classification_report = get_classification_report(true_labels,pred_labels,class_names)

# Print results
print("Accuracy score : ", accuracy_score)
print('--'*20)
print("Precision score : ", precision_score)
print('--'*20)
print("Recall score : ", recall_score)
print('--'*20)
print("F1 score : ", f1_score)
print('--'*20)
print("\nClassification report : ",classification_report)

# Uncertainty metrics
brier_score = get_brier_score(y_true=true_labels,y_pred_probs=probabilities)
print("\n Shape of true labels is : ",true_labels.shape)
expected_calibration_error = get_expected_calibration_error(y_true=true_labels,y_pred=probabilities)

print("Brier Score : ", round(brier_score,5))
print('--'*20)
print("Expected calibration error : ", round(expected_calibration_error,5))

# # Get best and worst predictions
# best_predictions,worst_predictions = get_best_worst_predictions(confidences)

# Plot confusion matrix
confusion_mat_fig = plot_confusion_matrix1(true_labels=true_labels,
                                            predicted_labels=pred_labels,
                                            class_names=class_names,
                                            results_path=save_path,
                                            plot_name=confusion_matrix_name)

# Plot the best and worst predictions.
best_fig = plot_predicted_images(predictions=pred_labels,
                                 confidences_pred_images=confidences,
                                 image_paths=image_paths,
                                 labels=true_labels,
                                 class_names=class_names,
                                 plot_preds=best_predictions,
                                 results_path=save_path,
                                 plot_name= best_preds_name
                                 )

worst_fig = plot_predicted_images(predictions=pred_labels,  
                                  confidences_pred_images=confidences,
                                  image_paths=image_paths,
                                  labels=true_labels,
                                  class_names=class_names,
                                  plot_preds=worst_predictions,
                                  results_path=save_path,
                                  plot_name= worst_preds_name
                                  )

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

# probabilities_file_path = str(save_path)+str(condition_name)+'_probabilities.csv'
# probabilities_df.to_csv(path_or_buf=probabilities_file_path)
# Calculate entropy values
if parameters['loss_function']=='Crossentropy':
    entropy_values = get_multinomial_entropy(probabilities)
    # Save probabilities to csv file
    probabilities_df = pd.DataFrame(probabilities)

elif parameters['loss_function']== 'Evidential':
    entropy_values = get_dirchlet_entropy(probabilities)
    # Save probabilities to csv file
    probabilities_df = pd.DataFrame(model_output)
    

# Save the results to csv files
results_dict = {
    "true_labels": true_labels,
    "pred_labels":pred_labels,
    "confidences":confidences,
    "image_paths": image_paths
}
# "entropy_values":entropy_values,
# probabilities_dict = {"probabilities":probabilities}
results_df = pd.DataFrame(results_dict)
dr_results_file_path = str(save_path)+str(condition_name)+'_entropy_results.csv'
# results_df.to_csv(path_or_buf=dr_results_file_path,sep=',')


# Final dataframe
final_results_file_path = str(save_path)+str(condition_name)+'_results.csv'
final_df = pd.concat([results_df,probabilities_df],axis=1)
final_df['ue'] = parameters['loss_function']
final_df['architecture'] = parameters['model_name']
final_df['constraint'] = data_set_type
final_df.to_csv(path_or_buf=final_results_file_path)

# Save model outputs to csv file
model_out_file_path = str(save_path)+str(condition_name)+'_model_output.csv'
model_out_df = pd.DataFrame(model_output)
model_out_df.to_csv(path_or_buf=model_out_file_path)
