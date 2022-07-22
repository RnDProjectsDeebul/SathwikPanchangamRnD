import torch
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


def save_architecture_txt(model,dir_path,filename):
        complete_file_name = os.path.join(dir_path, filename+"_arch.txt")
        with open(complete_file_name, "w") as f:
                f.write(str(model))
                f.close()

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device

def get_accuracy_score(true_labels,predicted_labels):
  return accuracy_score(true_labels,predicted_labels)

def get_precision_score(true_labels,predicted_labels):
  return precision_score(true_labels,predicted_labels,average='weighted')

def plot_confusion_matrix1(true_labels,predicted_labels,classes):
        fig, axs = plt.subplots(figsize=(20, 20))
        plot_confusion_matrix(true_labels,predicted_labels , ax=axs,normalize=True)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=0, fontsize=15)
        plt.yticks(tick_marks, classes, fontsize=15)
        plt.title('Confusion Matrix', fontsize=20)
        plt.savefig('results/confusion_matrix1.png')
        return fig

def plot_confusion_matrix2(true_labels,predicted_labels,classes):
        fig, axs = plt.subplots(figsize = (20,20))
        cf_matrix = confusion_matrix(true_labels, predicted_labels)
        df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes], columns = [i for i in classes])
        plt.figure(figsize = (12,7))
        sn.heatmap(df_cm, annot=True)
        plt.title('Confusion Matrix', fontsize=20)
        plt.savefig('results/confusion_matrix2.png')
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