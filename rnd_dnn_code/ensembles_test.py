# Import system modules
import os
import sys

# Import torch modules
import torch
from torchvision.models import resnet18,mobilenet_v2
from torch import nn

# Import custom modules
from data import load_data,load_test_data # load data: train-70,val-20,test-10 /// load_test_data:only test loader
import helpers
from helpers import get_model, test_for_ensembles,get_entropy_ensemble
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
data_dir = 'path/for/dataset/directory'
save_path = os.path.join(os.getcwd()+'/results/test_results/ensemble_results/')
models_path = os.path.join(os.getcwd()+'/results/training_results/ensemble_results/')

# Parameters
# Model name : Resnet18,Resnet9,Mobilenetv2 => Ensemble_models
# loss_functions: Crossentropy
# test_for : Ensembles
parameters = {'num_classes': 15,
              'batch_size': 64, 
              'model_name':'Ensemble_models',
              'loss_function': 'Crossentropy',
              'test_for':'Ensembles',
              'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}

# Set dataset name for the file name to save the best and worst preds plot 
dataset_name = 'robocup_normal_lighting'
condition_name = parameters['test_for']+'_'+dataset_name

best_preds_name = 'best_pred_'+str(condition_name)
worst_preds_name = 'worst_pred_'+str(condition_name)
confusion_matrix_name = 'confusion_matrix_'+str(condition_name)

# Initialing the neptune logger.
logger = False  
if logger:
    # run = neptune.init('Provide your neptune ai key')
    run = neptune.init(
    project="provide_neptune_project_name",
    api_token="provide_neptune_api_token",
    tags = ["Testing",dataset_name,str(parameters['test_for']),str(parameters['model_name'])],
    name= "Testing" + "-" + str(dataset_name) + "-" + str(parameters['model_name']) + "-" + str(parameters['test_for']),
    )
else:
    run = None

# Get the device CPU or GPU(CUDA)
device = parameters['device']

# Create the data loaders => change based on the dataset 
dataloader,class_names = load_data(data_dir=data_dir,batch_size=parameters['batch_size'],logger=run)
# dataloader,class_names = load_test_data(data_dir=data_dir,batch_size=parameters['batch_size'],logger=run)

test_loader = dataloader['test']
print("Number of test images : ",len(test_loader)*parameters['batch_size'])

# Create ensemble of models
model_dict = model = {
    "Resnet18":get_model('Resnet18',num_classes=parameters['num_classes']),
    "Mobilenetv2":get_model('Mobilenetv2',num_classes=parameters['num_classes']),
    "Mobilenetv3_small":get_model('Mobilenetv3_small',num_classes=parameters['num_classes']),
    "Resnet34":get_model('Resnet34',num_classes=parameters['num_classes']),
    }

model1 = model_dict['Resnet18']
model2 = model_dict['Resnet34']
model3 = model_dict['Mobilenetv2']
model4 = model_dict['Mobilenetv3_small']

model1_path = models_path + 'model1.pth'
model1.load_state_dict(torch.load(model1_path))

model2_path = models_path + 'model2.pth'
model2.load_state_dict(torch.load(model2_path))

model3_path = models_path + 'model3_name.pth'
model3.load_state_dict(torch.load(model3_path))

model4_path = models_path + 'model4_name.pth'
model4.load_state_dict(torch.load(model4_path))

e_models = [model1,model2,model3,model4] # Add models to the list

# Test for ensembles and get the results
ensemble_results = test_for_ensembles(dataloader=test_loader,
                    ensemble_models=e_models,
                    device=device,
                    loss_function=parameters['loss_function']
                )

# Seperate the results
true_labels = ensemble_results['true_labels']
pred_labels = ensemble_results['pred_labels']
probabilities = ensemble_results['probabilities']
model_output = ensemble_results['model_output']
images_list = ensemble_results['images_list']
confidences = ensemble_results['confidences']

# Calculate the entropy 
entropy_values = get_entropy_ensemble(probabilities)

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
expected_calibration_error = get_expected_calibration_error(y_true=true_labels,y_pred=probabilities)

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
                                 images=images_list,
                                 labels=true_labels,
                                 class_names=class_names,
                                 plot_preds=best_predictions,
                                 results_path=save_path,
                                 plot_name= best_preds_name
                                 )

worst_fig = plot_predicted_images(predictions=pred_labels,  
                                  confidences_pred_images=confidences,
                                  images=images_list,
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
    "entropy_values":entropy_values
}

results_df = pd.DataFrame(results_dict)
dr_results_file_path = str(save_path)+str(condition_name)+'_entropy_results.csv'
results_df.to_csv(path_or_buf=dr_results_file_path,sep=',')

# Save probabilities to csv file
probabilities_file_path = str(save_path)+str(condition_name)+'_probabilities.csv'
probabilities_df = pd.DataFrame(probabilities)
probabilities_df.to_csv(path_or_buf=probabilities_file_path)

# Save the dropout model output from farward passes to csv file
dr_out_file_path = str(save_path)+str(condition_name)+'_ensembles_model_output.txt'
with open(str(dr_out_file_path), 'w') as outfile:
    for slice_2d in model_output:
        np.savetxt(outfile, slice_2d)
