import torch
from torchvision.models import resnet18,mobilenet_v2
from torch import nn
import scipy.ndimage as nd
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix ,classification_report,accuracy_score,f1_score,precision_score,recall_score
from scikitplot.metrics import plot_confusion_matrix
import seaborn as sn
import glob
import json

def get_model(model_name,num_classes):
    # Creating a DNN model.
    if model_name == 'Resnet18':
        model = resnet18(weights = 'DEFAULT')
        assert model.__class__.__name__ == 'ResNet'
        # Set the output features of the last layer of the model to number of classes.
        model.fc = nn.Linear(in_features=512,out_features=num_classes)
    elif model_name == 'Mobilenetv2':
        model = mobilenet_v2(weights='DEFAULT')
        assert model.__class__.__name__ == 'MobileNetV2'
        # Set the output features of the last layer of the model to number of classes.
        model.classifier[1] = nn.Linear(in_features=model.last_channel,out_features=num_classes)
    elif model_name == 'Squeezenet':
        pass
    else:
        raise NameError

    return model


def test_one_epoch(dataloader,model,device,loss_function):
    print("Started testing")    
    confidence_list = []
    predicted_labels = []
    true_labels = []
    probabilities_list = []
    images = []
    labels_list = []

    with torch.no_grad():
        for batch_idx,(inputs,labels) in enumerate(dataloader):
            inputs,labels = inputs.to(device),labels.to(device)
            outputs = model(inputs)
            
            if loss_function == 'Crossentropy':
                outputs = torch.softmax(outputs,dim=1)
                confidences,predictions = torch.max(outputs,1)
            elif loss_function == 'Evidential':
                outputs = [round(i/np.sum(outputs),4) for i in outputs]
                confidences,predictions = torch.max(outputs,1)
            
            for image,label in zip(inputs,labels):
                images.append(image)
                labels_list.append(label)
            
            # Save
            probabilities_list.extend(outputs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())
            confidence_list.extend(confidences.cpu().numpy())
    confidence_array = np.array(confidence_list)
    predicted_labels = np.array(predicted_labels)
    true_labels = np.array(true_labels)
    probabilities_vector = np.array(probabilities_list)
    
    return true_labels,predicted_labels,confidence_array,probabilities_vector,images,labels_list

def get_best_worst_predictions(confidence_array):
    confidences_idxs_sorted = np.argsort(confidence_array)
    best_predictions = confidences_idxs_sorted[-9 : ]
    worst_predictions = confidences_idxs_sorted[:9]

    return best_predictions,worst_predictions

def plot_predicted_images(predictions,confidences_pred_images,images,labels,class_names,plot_preds,results_path,plot_name):

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

        # plt.savefig(results_path +'best_pred_evidential_dark_light.png' )
        plt.savefig(str(results_path) + str(plot_name) +'.png' )
    return fig


def get_brier_score(y_true, y_pred_probs):
  return 1 + (np.sum(y_pred_probs ** 2) - 2 * np.sum(y_pred_probs[np.arange(y_pred_probs.shape[0]), y_true])) / y_true.shape[0]


def get_expected_calibration_error(y_true, y_pred, num_bins=15):
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


def one_hot_embedding(labels, num_classes):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    return y[labels]

def save_architecture_txt(model,dir_path,filename):
        complete_file_name = os.path.join(dir_path, filename+"_arch.txt")
        with open(complete_file_name, "w") as f:
                f.write(str(model))
                f.close()

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def get_accuracy_score(true_labels,predicted_labels):
  return accuracy_score(true_labels,predicted_labels)

def get_precision_score(true_labels,predicted_labels):
  return precision_score(true_labels,predicted_labels,average='weighted')

def plot_confusion_matrix1(true_labels,predicted_labels,class_names,results_path,plot_name):
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

def get_classification_report(true_labels,predicted_labels,classes):
        class_report = classification_report(true_labels, predicted_labels,target_names=classes)
        return class_report

def get_f1_score(true_labels,predicted_labels):
        f1score = f1_score(true_labels, predicted_labels,average='weighted')
        return f1score

def get_recall_score(true_labels,predicted_labels):
        recall = recall_score(true_labels, predicted_labels,average='weighted')
        return recall

def plot_losses(train_loss,valid_loss,criterion_name):
        fig, ax = plt.subplots(figsize=(20,20))
        ax.plot(train_loss,label='Training Loss')
        ax.plot(valid_loss,label='Validation Loss')
        plt.xlabel('# Epoch', fontsize=15)
        plt.ylabel(str(criterion_name),fontsize=15)
        
        plt.title('Loss Plot')
        plt.legend()
        fig.savefig('/home/sathwikpanchngam/rnd/github_projects/SathwikPanchangamRnD/Deeplearning_pipeline/test_results/losses_plot.png')
        return fig
        # Add saving path as an input parameter to the function
        

def plot_accuracies(train_acc,valid_acc):
        fig, ax = plt.subplots(figsize=(20,20))
        ax.plot(train_acc,label='Training accuracy')
        ax.plot(valid_acc,label='Validation accuracy')
        plt.xlabel('# Epoch')
        plt.ylabel('Accuracy')
        plt.title("Accuracy Plot")
        plt.legend()
        fig.savefig('/home/sathwikpanchngam/rnd/github_projects/SathwikPanchangamRnD/Deeplearning_pipeline/test_results/accuracies_plot.png')
        return fig
        # Add saving path as an input parameter to the function

def imshow(img):
  ''' function to show image '''
  img = img / 2 + 0.5 # unnormalize
  npimg = img.numpy() # convert to numpy objects
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()


# Early stopping
def early_stopping(train_loss,validation_loss,min_delta,tolerance):
        counter = 0
        if (validation_loss-train_loss) > min_delta:
                counter += 1
                if counter>=tolerance:
                        return True
