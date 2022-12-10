# import system modules
import os
import sys
import glob
import json

# import torch modules
import torch
from torchvision.models import resnet18,mobilenet_v2
from torch import nn
import torch.nn.functional as F

# import other modules
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import classification_report,accuracy_score,f1_score,precision_score,recall_score
from scikitplot.metrics import plot_confusion_matrix

# helper function for dropout testing
# Dropout test code.
def test_for_dropout(dataloader,model,device,loss_function,num_farward_passes):
  """ Function for testing/inference of dropout model
  """
  # Results variables
  images_list = []
  # probabilitites_list = [[] for _ in range(num_farward_passes)] # => output_predictions
  confidences_list = []

  true_labels_list = []
  pred_labels_list = []
  # Dropout variables
  output_predictions = [[] for _ in range(num_farward_passes)]
  # Dropout
  if loss_function=='Crossentropy':
      print("Started testing for dropout")
      # Set the model to training mode to enable dropout
      # Begin testing
      with torch.no_grad():

          for batch_idx,(inputs,labels) in enumerate(dataloader):
              for image,label in zip(inputs,labels):
                  images_list.append(image)
                  true_labels_list.append(label)
              inputs,labels = inputs.to(device),labels.to(device)

              for fwd_pass in range(num_farward_passes):

                  # Set the model to train mode, i.e enable dropout
                  # Change the code based on timm models
                  model.train()
                  model.to(device=device)

                  # Get the logits
                  output = model(inputs)
                  output = torch.softmax(output,dim=1)

                  # Save probabilities and multinomial output both are same here.
                  np_output = output.detach().cpu().numpy()

                  # Store the results from farward passes
                  if not output_predictions[fwd_pass] != []:
                      output_predictions[fwd_pass] = np_output
                      #print("First values")
                  else:
                      output_predictions[fwd_pass] = np.vstack((output_predictions[fwd_pass], np_output))
                      #print("Stacking the values")

      # Compute dropout results           
      output_predictions = np.stack(output_predictions,axis=1)
      probabilities = np.mean(output_predictions,axis=1)
      confidences_list.extend(np.max(probabilities,axis=1))
      pred_labels_list.extend(np.argmax(probabilities,axis=1))

      # return dropout results
      dropout_results_dict = {
          "true_labels": np.array(true_labels_list),
          "pred_labels": np.array(pred_labels_list),
          "probabilities":probabilities,
          "model_output":output_predictions,
          "images_list": images_list,
          "confidences": np.array(confidences_list),
          }

      return dropout_results_dict

# helper funciton for ensemble testing
def test_for_ensembles(dataloader,ensemble_models,device,loss_function):
  """ Function for testing/inference of ensemble models
  """
  # Results variables
  ensemble_image_list = []
  ensemble_true_labels = []
  ensemble_pred_labels = []
  ensemble_confidences = []

  # Variable for ensemble results.    
  ensemble_outputs_list = [[] for _ in range(len(ensemble_models))]

  # Test for Ensembles
  if loss_function == 'Crossentropy':
      print("Started Ensemble testing")
      # Begin testing
      with torch.no_grad():
          for batch_idx,(inputs,labels) in enumerate(dataloader):
              #print("batch_idx",batch_idx)
              for image,label in zip(inputs,labels):
                  ensemble_image_list.append(image)
                  ensemble_true_labels.append(label)

              # Set the inputs and models to cuda
              inputs,labels = inputs.to(device),labels.to(device) 

              for idx_model,model in enumerate(ensemble_models):

                  # Set the model to evaluation mode
                  model.to(device=device)
                  model.eval()

                  # Get the logits
                  output = model(inputs)
                  output = torch.softmax(output,dim=1)

                  # Save probabilities and multinomial output both are same here.
                  np_output = output.detach().cpu().numpy()

                  if not ensemble_outputs_list[idx_model] != []:
                      ensemble_outputs_list[idx_model] = np_output
                     # print(ensemble_outputs_list[idx_model].shape)
                  else:
                      #print("idx_model",idx_model)
                      ensemble_outputs_list[idx_model] = np.vstack((ensemble_outputs_list[idx_model] , np_output))

  # Compute ensemble final results
  ensemble_outputs_array = np.stack(np.array(ensemble_outputs_list))
  ensemble_probabilities = np.mean(ensemble_outputs_array,axis=0)
  ensemble_confidences.extend(np.max(ensemble_probabilities,axis=1))
  ensemble_pred_labels.extend(np.argmax(ensemble_probabilities,axis=1))

  # return ensemble results
  ensemble_results_dict = {
          "true_labels": np.array(ensemble_true_labels),
          "pred_labels": np.array(ensemble_pred_labels),
          "probabilities":ensemble_probabilities,
          "model_output":ensemble_outputs_array,
          "images_list": ensemble_image_list,
          "confidences": np.array(ensemble_confidences)
          }

  return ensemble_results_dict

# helper functions for plotting training results
def plot_accuracies(train_acc,valid_acc,save_path):
  """ Function to plot accuracy curve
  """
  fig, ax = plt.subplots(figsize=(20,20))
  ax.plot(train_acc,label='Training accuracy')
  ax.plot(valid_acc,label='Validation accuracy')
  plt.xlabel('# Epoch')
  plt.ylabel('Accuracy')
  plt.title("Accuracy Plot")
  plt.legend()
  fig.savefig(str(save_path)+'/accuracies_plot.png')
  return fig
      
def plot_losses(train_loss,valid_loss,criterion_name,save_path):
  """ Function to plot losses
  """
  fig, ax = plt.subplots(figsize=(20,20))
  ax.plot(train_loss,label='Training Loss')
  ax.plot(valid_loss,label='Validation Loss')
  plt.xlabel('# Epoch', fontsize=15)
  plt.ylabel(str(criterion_name),fontsize=15)
  plt.title('Loss Plot')
  plt.legend()
  fig.savefig(str(save_path)+'/losses_plot.png')
  return fig

# helpers for test results
def get_best_worst_predictions(confidence_array):
  """Function to compute best and worst predictions
  """
  confidences_idxs_sorted = np.argsort(confidence_array)
  best_predictions = confidences_idxs_sorted[-9 : ]
  worst_predictions = confidences_idxs_sorted[:9]
  return best_predictions,worst_predictions

# helper functions for plotting test results
def imshow(img):
  """ function to show image 
  """
  img = img / 2 + 0.5 # unnormalize
  npimg = img.numpy() # convert to numpy objects
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()
  
def plot_confusion_matrix1(true_labels,predicted_labels,class_names,results_path,plot_name):
  """Function to plot confusion matrix based on scikitplot metrics
  """
  fig, axs = plt.subplots(figsize=(20, 20))
  plot_confusion_matrix(true_labels,predicted_labels , ax=axs,normalize=True)
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45, fontsize=15)
  plt.yticks(tick_marks, class_names, rotation=45,fontsize=15)
  plt.title('Confusion Matrix', fontsize=20)
  plt.savefig(str(results_path)+str(plot_name)+'.png')
  plt.xlabel("Predicted label",fontsize=15)
  plt.ylabel("True label",fontsize=15)
  return fig

def plot_predicted_images(predictions,confidences_pred_images,images,labels,class_names,plot_preds,results_path,plot_name):
  """ Function to plot the predicted images
  """
  rows = 3
  columns = 3
  fig = plt.figure(figsize=(20,20))

  for idx,i in enumerate(plot_preds):
      idx = idx+1
      img_ = images[i].cpu().detach()
      label_true = labels[i]
      label_pred = predictions[i]
      confidence = confidences_pred_images [i]

      fig.add_subplot(rows, columns, idx)
      # showing image
      plt.imshow(img_.permute(1,2,0))
      plt.title("True_label : "+ str(class_names[label_true]))
      plt.xlabel("Confidance : " + str(round(confidence,4)))
      plt.ylabel("Pred_label : " + str(class_names[label_pred]))

      plt.rc('axes', titlesize=20)
      plt.rc('axes', labelsize=20)
      plt.rc('xtick', labelsize=0)
      plt.rc('ytick', labelsize=0)
      plt.savefig(str(results_path) + str(plot_name) +'.png' )
  return fig

# helpers for classification metrics
def get_accuracy_score(true_labels,predicted_labels):
  """Function to compute accuracy score based on sklearn metrics
  """
  return accuracy_score(true_labels,predicted_labels)

def get_precision_score(true_labels,predicted_labels):
  """Function to compute precision score based on sklearn metrics
  """
  return precision_score(true_labels,predicted_labels,average='weighted')

def get_f1_score(true_labels,predicted_labels):
  """Function to compute f1 score based on sklearn metrics
  """
  return f1_score(true_labels, predicted_labels,average='weighted')

def get_recall_score(true_labels,predicted_labels):
  """Function to compute recall score based on sklearn metrics
  """
  return recall_score(true_labels, predicted_labels,average='weighted')

def get_classification_report(true_labels,predicted_labels,class_names):
  """Function to compute classification report based on sklearn metrics
  """
  return classification_report(true_labels, predicted_labels,target_names=class_names)

# helpers for calibration metrics
def get_brier_score(y_true, y_pred_probs):
  """ Function to compute brier score
  """
  return 1 + (np.sum(y_pred_probs ** 2) - 2 * np.sum(y_pred_probs[np.arange(y_pred_probs.shape[0]), y_true])) / y_true.shape[0]

def get_expected_calibration_error(y_true, y_pred, num_bins=15):
  """ Function to compute expected calibration error
  """
  pred_y = np.argmax(y_pred, axis=-1)
  correct = (pred_y == y_true).astype(np.float32)
  prob_y = np.max(y_pred, axis=-1)

  b = np.linspace(start=0, stop=1.0, num=num_bins)
  bins = np.digitize(prob_y, bins=b, right=True)

  o = 0
  for b in range(num_bins):
    mask = bins == b
    if np.any(mask):
        o += np.abs(np.sum(correct[mask] - prob_y[mask]))
  
  x = np.sum(prob_y[mask]/prob_y.shape[0])
  y = np.sum(y_true[mask] /y_true.shape[0])
  return o / y_pred.shape[0]


