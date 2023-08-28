# Import system modules
import os
import sys
# Import torch modules
import torch
from torchvision.models import resnet18,mobilenet_v2,squeezenet1_1 # Import model from timm library
from torch import nn
from timm import models

# Import custom modules
from data import load_data,load_test_data,load_synthetic_data # load data: train-70,val-20,test-10 /// load_test_data:only test loader
import helpers
from helpers import get_model, test_for_dropout, get_entropy_dropout
from helpers import get_accuracy_score,get_precision_score,get_recall_score,get_f1_score,get_classification_report
from helpers import get_brier_score,get_expected_calibration_error
from helpers import get_best_worst_predictions, plot_predicted_images
from helpers import plot_confusion_matrix1

# Import other libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import logger
import neptune.new as neptune

# Set the required paths
data_dir = '/path/test_data/'
models_path = '/path/trained_models/'
save_path = '/path/test_results/'

# Parameters
# Model name : Resnet18
# loss_functions: Crossentropy
# test_for : Dropout
parameters = {'num_classes': 35,
              'batch_size': 64, 
              'model_name':'Resnet18',
              'loss_function': 'Crossentropy',
              'test_for':'Dropout',
              'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
              'logger':False,
             }

# Set dataset name for the file name to save the best and worst preds plot 
dataset_name = 'blur'

print("Testing for : ",dataset_name)
condition_name = str(dataset_name)+'_'+parameters['test_for']

# Get the model/create the dnn model
model = models.create_model('resnet18',pretrained=False,num_classes=parameters['num_classes'],drop_rate=0.3)

# Load the best saved model from the training results.
model_path = models_path + 'normal_Crossentropy_Dropout_model.pth'
model.load_state_dict(torch.load(model_path))
# print(model)


# Names for results files
best_preds_name = 'best_pred_'+str(condition_name)
worst_preds_name = 'worst_pred_'+str(condition_name)
confusion_matrix_name = 'confusion_matrix_'+str(condition_name)


# Initialing the neptune logger.
if parameters['logger']:
    # run = neptune.init('Provide your neptune ai key')
    run = neptune.init(
    project="provide experiment name check in neptune ai",
    api_token="provide api key",
    tags = ["Testing",dataset_name,str(parameters['test_for']),str(parameters['model_name'])],
    name= "Testing" + "-" + str(dataset_name) + "-" + str(parameters['model_name']) + "-" + str(parameters['test_for']),
    )
else:
    run = None

# Get the device CPU or GPU(CUDA)
device = parameters['device']

# Create the data loaders => change based on the dataset 
if dataset_name == 'normal':
    # Load the dataloader for synthetic data.
    dataloader,class_names = load_synthetic_data(data_dir=data_dir,batch_size=parameters['batch_size'],logger=run)
else:
    # Load the dataloader for synthetic data.
    dataloader,class_names = load_test_data(data_dir=data_dir,batch_size=parameters['batch_size'],logger=run)


test_loader = dataloader['val']
print("Number of test images : ",len(test_loader)*parameters['batch_size'])



# Test for dropout and get results
dropout_results = test_for_dropout(model=model,
                         dataloader=test_loader,
                         device=device,
                         loss_function=parameters['loss_function'],
                         num_farward_passes=5
                         )


# Seperate the results
true_labels=dropout_results["true_labels"]
pred_labels=dropout_results["pred_labels"]
probabilities=dropout_results["probabilities"]
model_output = dropout_results['model_output']
images_list=dropout_results["images_list"]
confidences=dropout_results["confidences"]
image_paths = dropout_results['image_paths']

# Calculate the entropy 
entropy_values = get_entropy_dropout(stacked_probabilities=model_output)

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
brier_score = get_brier_score(y_true=true_labels, y_pred_probs=probabilities)
expected_calibration_error = get_expected_calibration_error(y_true= true_labels, y_pred=probabilities)

print("Brier Score : ", brier_score)
print('--'*20)
print("Expected calibration error : ", expected_calibration_error)

# Get best and worst predictions
best_predictions,worst_predictions = get_best_worst_predictions(confidences)

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

# Log the metrics to neptue ai
if run !=None:
    # Logging the results.

    # log model parameters
    run['config/hyperparameters'] = parameters
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
    
# Save the results to csv files
results_dict = {
    "true_labels": true_labels,
    "pred_labels":pred_labels,
    "confidences":confidences,
    "image_paths":image_paths
}
results_df = pd.DataFrame(results_dict)
# dr_results_file_path = str(save_path)+str(condition_name)+'_entropy_results.csv'
# results_df.to_csv(path_or_buf=dr_results_file_path,sep=',')

# Save probabilities to csv file
# probabilities_file_path = str(save_path)+str(condition_name)+'_probabilities.csv'
probabilities_df = pd.DataFrame(probabilities)
# probabilities_df.to_csv(path_or_buf=probabilities_file_path)

# Entropy results
entropy_df = {"entropy":entropy_values}
entropy_df = pd.DataFrame(entropy_df)

# Final dataframe
final_results_file_path = str(save_path)+str(condition_name)+'_results.csv'
final_df = pd.concat([results_df, probabilities_df, entropy_df], axis=1)
final_df['ue'] = parameters['test_for']
final_df['architecture'] = parameters['test_for']
final_df['constraint'] = dataset_name
final_df.to_csv(path_or_buf=final_results_file_path)

# Save the dropout model output from farward passes to csv file
dr_out_file_path = str(save_path)+str(condition_name)+'_dropout_model_output.txt'
with open(str(dr_out_file_path), 'w') as outfile:
    for slice_2d in model_output:
        np.savetxt(outfile, slice_2d)

