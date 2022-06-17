import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix ,classification_report,accuracy_score,f1_score,precision_score,recall_score
from scikitplot.metrics import plot_confusion_matrix
import seaborn as sn
import glob



class Metrics():
    """
    Creates helper functions of the classification metrics.

    Parameters
    ----------
    y_target: 
        True labels of the images.
    y_predict: 
        Predicted labels of the images.
    classes: list(str)
        Class names of the images.
    train_loss:
        List of train losses for all epochs.
    test_loss:
        List of test losses for all epochs.
    """
    def __init__(self,y_target,y_predict, classes,train_loss,test_loss):

        self.predicts = y_predict

        self.true_labels = y_target
        self.predict_labels = y_predict
        self.classes = classes
        self.train_loss = train_loss
        self.test_loss = test_loss

    def get_accuracy(self):
        accuracy = accuracy_score(self.true_labels, self.predict_labels)
        return accuracy
        
    def get_precision_score(self):
        precision = precision_score(self.true_labels, self.predict_labels,average='weighted')
        return precision

    def plot_confusion_matrix1(self):
        fig, axs = plt.subplots(figsize=(20, 20))
        plot_confusion_matrix(self.true_labels, self.predict_labels , ax=axs,normalize=True)
        tick_marks = np.arange(len(self.classes))
        plt.xticks(tick_marks, self.classes, rotation=0, fontsize=15)
        plt.yticks(tick_marks, self.classes, fontsize=15)
        plt.title('Confusion Matrix', fontsize=20)
        plt.savefig('results/confusion_matrix1.png')
        return fig

    def plot_confusion_matrix2(self):
        fig, axs = plt.subplots(figsize = (20,20))
        cf_matrix = confusion_matrix(self.true_labels, self.predict_labels)
        df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in self.classes], columns = [i for i in self.classes])
        plt.figure(figsize = (12,7))
        sn.heatmap(df_cm, annot=True)
        plt.title('Confusion Matrix', fontsize=20)
        plt.savefig('results/confusion_matrix2.png')
        return fig

    def get_classification_report(self):
        class_report = classification_report(self.true_labels, self.predict_labels,target_names=self.classes)
        return class_report

    def get_f1_score(self):
        f1score = f1_score(self.true_labels, self.predict_labels,average='weighted')
        return f1score

    def get_recall_score(self):
        recall = recall_score(self.true_labels, self.predict_labels,average='weighted')
        return recall
    
    # Function to plot train loss
    def plot_train_loss(self):
        fig, axs = plt.subplots(figsize=(20, 20))
        axs.plot(self.train_loss, label='Training Loss')
        plt.ylabel('Loss', fontsize=15)
        plt.xlabel('# Epoch', fontsize=15)
        plt.title('Training Loss')
        plt.legend()
        plt.savefig('results/train_loss.png')
        return fig
    # Function to plot test loss
    def plot_test_loss(self):
        fig, axs = plt.subplots(figsize=(20, 20))
        axs.plot(self.test_loss, label='Test Loss')
        plt.ylabel('Loss', fontsize=15)
        plt.xlabel('# Epoch', fontsize=15)
        plt.title('Test Loss')
        plt.legend()
        plt.savefig('results/test_loss.png')
        return fig
    # Function to show some of the images form each class.

    def show_images(self, image):
        fig, axs = plt.figure(figsize=(25,4))
        for idx in np.arange(5):
            ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks =[])
            image = image/2 + 0.5
            plt.imshow(np.transpose(image, (1,2,0)))
            ax.set_title(self.classes[self.true_labels[idx]])
        return fig
    
    # Helper function to un normalize and display the image
    def imshow(self,img):
        img = img/2 + 0.5 #Unnormalize
        plt.imshow(np.transpose(img, (1,2,0))) # Convert from tensor to image

    def show_images_1(self,root,trainloader):
        classes = []
        for path in glob.glob(root + '/*'):
            classes.append(path.split('/')[-1])

        # Obtaining one batch of training images
        dataiter = iter(trainloader)
        images, labels = dataiter.next()
        images = images.numpy()
        fig = plt.figure(figsize= (12,12))
        # Display 20 images
        for idx in np.arange(15):
            ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks =[])
            self.imshow(images[idx])
            ax.set_title(classes[labels[idx]])

        plt.savefig("./images/exapmle1.png")