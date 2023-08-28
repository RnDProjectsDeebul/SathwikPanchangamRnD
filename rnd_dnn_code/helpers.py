import torch
import torchvision.models
from torchvision.models import resnet18,mobilenet_v2,resnet34,mobilenet_v3_small
# from torchvision.models import mobilenet_v3_small
from torch import nn
import torch.nn.functional as F
import scipy.ndimage as nd
import cv2
from torchvision.datasets import ImageFolder
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import dirichlet, multinomial
from sklearn.metrics import classification_report,accuracy_score,f1_score,precision_score,recall_score
from scikitplot.metrics import plot_confusion_matrix
import seaborn as sns
import glob
import json
import sys
import matplotlib.image as mpimg

def load_csv_files_as_numpy(dir_path,test_conditions):
    test_conditions = test_conditions
    print("Number of test conditions",len(test_conditions))
    
    for loss in ['Evidential']:
        for model in ['Resnet18']:
            for condition in test_conditions:
                # Get the file path
                file_path = dir_path+condition+'_'+loss+'_'+model+'_results.csv'
                # Read the csv files
                evidential_results = np.genfromtxt(file_path, delimiter=',')
                evidential_results = np.delete(evidential_results,0,axis=0)
                evidential_results = np.delete(evidential_results,[0,19,20,21],axis=1)
                print("changed alpha values")
                alpha_values = evidential_results[0:,3:18]
                alpha_values2= np.array([np.array([alpha/np.sum(vector) for j,alpha in enumerate(vector)]) for i,vector in enumerate(alpha_values)])
                evidential_results[0:,3:18] = alpha_values2
    for loss in ['Crossentropy']:
        for model in ['Resnet18']:
            for condition in test_conditions:
                # Get the file path
                file_path = dir_path+condition+'_'+loss+'_'+model+'_results.csv'
                # Read the csv files
                crossentropy_results = np.genfromtxt(file_path, delimiter=',')
                crossentropy_results = np.delete(crossentropy_results,0,axis=0)
                crossentropy_results = np.delete(crossentropy_results,[0,19,20,21],axis=1)
    
    for model in ['Dropout']:
        for condition in test_conditions:
            # Get the file path
            file_path = dir_path+condition+'_'+model+'_results.csv'
            # Read the csv files
            dropuout_results = np.genfromtxt(file_path, delimiter=',')
            dropuout_results = np.delete(dropuout_results,0,axis=0)
            dropuout_results = np.delete(dropuout_results,[0,19,20,21],axis=1)
    for model in ['Ensembles']:
        for condition in test_conditions:
            # Get the file path
            file_path = dir_path+condition+'_'+model+'_results.csv'
            # Read the csv files
            ensembles_results = np.genfromtxt(file_path, delimiter=',')
            ensembles_results = np.delete(ensembles_results,0,axis=0)
            ensembles_results = np.delete(ensembles_results,[0,19,20,21],axis=1)
    ue_resutls = {"Crossentropy":crossentropy_results, 
                  "Evidential":evidential_results,
                  "Dropout":dropuout_results,
                  "Ensembles":ensembles_results}
    
    return ue_resutls

# Helper functions for entropy plots
def plot_entropy_constraints(data_df,test_constraints,save_path,file_name):
    """ Dont inclued normal in test constraints list
        Change the constraints manually for now use this function for plotting entropy
    """
    plt.figure(figsize=(10,10))
    test_constraints.append('normal')
    sub = data_df[data_df['constraint'].isin(test_constraints)]
    d = {True: 'correct', False: 'incorrect'}

    sub['is_prediction_correct'] = sub['is_prediction_correct'].replace(d)
    sub['constraint_separated'] = sub['is_prediction_correct'] + "_" + sub['constraint']

    sub = sub[sub['constraint_separated'].isin(['correct_normal', 'correct_near', 'correct_far','incorrect_normal'])]


    g = sns.catplot(
    data=sub, y="entropy", x="constraint_separated", col="ue_method",
    kind="box", height=5, aspect=1, row_order = ['correct_normal','correct_near','correct_far', 'incorrect_normal'],
    )
    plt.show()
    print ("------------------------------------------------------")



def plot_entropy_correct_incorrect(data_df,
                                    test_constraints,
                                    save_path,
                                    file_name):
    """returns a fig with box plot for the correct and 
       incorrect predictions of all ue_methods for the 
       provided test constraints.
    """
    plt.figure(figsize=(10,10))

    sub = data_df[data_df['constraint'].isin(test_constraints)]
    d = {True: 'correct', False: 'incorrect'}
    sub['is_prediction_correct'] = sub['is_prediction_correct'].replace(d)
    sub = sub[sub['constraint'].isin(test_constraints)]
    g = sns.catplot(
                    data=sub, y="entropy", x="is_prediction_correct",
                    col="ue", hue="constraint",
                    kind="box",height=5, aspect=1,
                )
    g.set_titles(col_template="{col_name} Textures constraint")
    if file_name != None:
        plt.savefig(save_path+file_name)
    plt.show()
    return g

def entropy_values_correct_wrong_predictions(data_df):
    """ Prints the mean entropy value for all ue_methods for all constraints.
    """
    for ue_method in data_df.ue_method.unique():
        subset = data_df[(data_df['ue_method'] == ue_method)]
    for model in subset.architecture.unique():
        subset_model = subset[subset['architecture']==model]
        for condition in subset_model.constraint.unique():
            print (ue_method, model, condition)
            subset_cond = subset_model[subset_model['constraint']==condition]
            print ("Correct : ", subset_cond[subset_cond['is_prediction_correct']==True]['entropy'].mean())
            print ("Wrong : ", subset_cond[subset_cond['is_prediction_correct']==False]['entropy'].mean())
    return None



def read_csv_files(file_path):
    df = pd.read_csv(file_path)
    return df

def calculate_multinomial_entropy( p_values ):
    return multinomial(1, p_values).entropy()

def calculate_dirchlet_entropy(alpha_values):
  return dirichlet(alpha_values).entropy()

# Helper functions for combining the results into data frame.
def combine_experiment_results_to_data_frame(results_path:str,test_conditions):
    data_dir = results_path
    # First combine all normal conditions
    test_conditions = test_conditions
    print("Number of test conditions",len(test_conditions))
    
    df0 = pd.DataFrame()
    for loss in ['Evidential']:
        for model in ['Resnet18']:
            for condition in test_conditions:
                # Get the file path
                file_path = data_dir+condition+'_'+loss+'_'+model+'_results.csv'
                # Read the csv files
                model_out_df = read_csv_files(file_path)
                # Drop unnecessary columns
                model_out_df = model_out_df.drop(columns=['Unnamed: 0'])
                model_out = model_out_df.to_numpy()
                alpha_values = model_out[0:,3:18]
                alpha_values= np.array([i/np.sum(i) for i in alpha_values])
                model_out[0:,3:18] = alpha_values
                model_out_df = pd.DataFrame(model_out,columns=['true_labels', 'pred_labels', 'confidences', '0', '1', '2', '3', '4',
                                                                '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', 'ue',
                                                                'architecture', 'constraint'])
                if loss=='Evidential':
                    model_out_df['entropy']=model_out_df.apply(lambda row: calculate_multinomial_entropy( [row['0'], row['1'], row['2'],
                                    row['3'], row['4'], row['5'], row['6'],row['7'], row['8'], row['9'], row['10'], row['11'], 
                                    row['12'], row['13'],row['14']]), axis=1)
                df0 = pd.concat([model_out_df, df0])
    
    df1 = pd.DataFrame()
    for loss in ['Crossentropy']:
        for model in ['Resnet18']:
            for condition in test_conditions:
                # Get the file path
                file_path = data_dir+condition+'_'+loss+'_'+model+'_results.csv'
                # Read the csv files
                model_out_df = read_csv_files(file_path)
                # Drop unnecessary columns
                model_out_df = model_out_df.drop(columns=['Unnamed: 0'])
                if loss == 'Crossentropy':
                    model_out_df['entropy']=model_out_df.apply(lambda row: calculate_multinomial_entropy( [row['0'], row['1'], row['2'],
                                    row['3'], row['4'], row['5'], row['6'],row['7'], row['8'], row['9'], row['10'], row['11'], 
                                    row['12'], row['13'],row['14']]), axis=1)
                df1 = pd.concat([model_out_df, df1])

    df2 = pd.DataFrame()
    # for loss in ['Dropout']:
    for model in ['Dropout']:
        for condition in test_conditions:
            # Get the file path
            file_path = data_dir+condition+'_'+model+'_results.csv'
            # Read the csv files
            model_out_df = read_csv_files(file_path)

            # Drop unnecessary columns
            model_out_df = model_out_df.drop(columns=['Unnamed: 0'])
            df2 = pd.concat([model_out_df, df2])

    df3 = pd.DataFrame()
    # for loss in ['Ensembles']:
    for model in ['Ensembles']:
        for condition in test_conditions:
            # Get the file path
            file_path = data_dir+condition+'_'+model+'_results.csv'
            # Read the csv files
            model_out_df = read_csv_files(file_path)
            # Drop unnecessary columns
            model_out_df = model_out_df.drop(columns=['Unnamed: 0'])
            df3 = pd.concat([model_out_df, df3])

    df = pd.concat([df0,df1,df2,df3])
    df = df.astype({'entropy': 'float64'})
    df['is_prediction_correct'] = df['true_labels'] == df['pred_labels']


    return df

# Ensembles test code.
def test_for_ensembles(dataloader,ensemble_models,device,loss_function):

    # Results variables
    ensemble_image_list = []
    ensemble_true_labels = []
    ensemble_pred_labels = []
    ensemble_confidences = []
    image_paths = []

    # Variable for ensemble results.    
    ensemble_outputs_list = [[] for _ in range(len(ensemble_models))]

    # Test for Ensembles
    if loss_function == 'Crossentropy':
        print("Started Ensemble testing")
        # Begin testing
        with torch.no_grad():
            for batch_idx,(inputs,labels,images_paths) in enumerate(dataloader):
                #print("batch_idx",batch_idx)
                for image,label,img_path in zip(inputs,labels,images_paths):
                    ensemble_image_list.append(image)
                    ensemble_true_labels.append(label)
                    image_paths.append(img_path)
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
    ensemble_outputs_array = np.stack(np.array(ensemble_outputs_list),axis=1)
    ensemble_probabilities = np.mean(ensemble_outputs_array,axis=1)
    ensemble_probabilities = [i/np.sum(i) for i in ensemble_probabilities]
    ensemble_confidences.extend(np.max(ensemble_probabilities,axis=1))
    ensemble_pred_labels.extend(np.argmax(ensemble_probabilities,axis=1))

    # return ensemble results
    ensemble_results_dict = {
            "true_labels": np.array(ensemble_true_labels),
            "pred_labels": np.array(ensemble_pred_labels),
            "probabilities":np.array(ensemble_probabilities),
            "model_output":ensemble_outputs_array,
            "images_list": ensemble_image_list,
            "confidences": np.array(ensemble_confidences),
            "image_paths": image_paths
            }

    return ensemble_results_dict

def get_multinomial_entropy( p_values ):
    entropy_values = []
    for i in p_values:
        entropy_values.append(multinomial(1, i).entropy())
    return entropy_values

def calculate_dirchlet_entropy(alpha_values):
  return dirichlet(alpha_values).entropy()

def get_dirchlet_entropy(alpha_values):
    entropy_values = []
    for i in alpha_values:
        entropy_values.append(calculate_dirchlet_entropy(i))
    return entropy_values

def predictive_entropy(predictions):
    epsilon = sys.float_info.min
    predictive_entropy = -np.sum( np.mean(predictions, axis=0) * np.log(np.mean(predictions, axis=0) + epsilon),
            axis=-1)
    return predictive_entropy


def get_entropy_dropout(stacked_probabilities):
    entropy_values = []
    for img_prob in stacked_probabilities:
        entropy_values.append(predictive_entropy(img_prob))
    return np.array(entropy_values)

def get_entropy_ensemble(stacked_probabilities):
    entropy_values = []
    for img_prob in stacked_probabilities:
        entropy_values.append(predictive_entropy(img_prob))
    return np.array(entropy_values)

def relu_evidence(y):
    return F.relu(y)

def enable_dropout(model,device):
    model.to(device=device)
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
            print("Dropout enabled")

# Dropout test code.
def test_for_dropout(dataloader,model,device,loss_function,num_farward_passes):
    
    # Results variables
    images_list = []
    # probabilitites_list = [[] for _ in range(num_farward_passes)] # => output_predictions
    confidences_list = []
    image_paths = []
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

            for batch_idx,(inputs,labels,images_paths) in enumerate(dataloader):
                for image,label,img_path in zip(inputs,labels,images_paths):
                    images_list.append(image)
                    true_labels_list.append(label)
                    image_paths.append(img_path)
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
        probabilities = [i/np.sum(i) for i in probabilities]
        confidences_list.extend(np.max(probabilities,axis=1))
        pred_labels_list.extend(np.argmax(probabilities,axis=1))

        # return dropout results
        dropout_results_dict = {
            "true_labels": np.array(true_labels_list),
            "pred_labels": np.array(pred_labels_list),
            "probabilities":np.array(probabilities),
            "model_output":output_predictions,
            "images_list": images_list,
            "confidences": np.array(confidences_list),
            "image_paths": image_paths
            }

        return dropout_results_dict

def get_model(model_name,num_classes,weights):
    """ Function for creating DNN model from torchvision models
    """
    # Creating a DNN model.
    if model_name == 'Resnet18':
        model = resnet18(pretrained = weights)
        assert model.__class__.__name__ == 'ResNet'
        # Set the output features of the last layer of the model to number of classes.
        model.fc = nn.Linear(in_features=512,out_features=num_classes)
    elif model_name == 'Mobilenetv2':
        model = mobilenet_v2(weights=weights)
        assert model.__class__.__name__ == 'MobileNetV2'
        # Set the output features of the last layer of the model to number of classes.
        model.classifier[1] = nn.Linear(in_features=1280,out_features=num_classes)
    elif model_name == 'Resnet34':
        model = resnet34(weights=weights)
        assert model.__class__.__name__ == 'ResNet'
        # Set the output features of the last layer of the model to number of classes.
        model.fc = nn.Linear(in_features=512,out_features=num_classes)
    elif model_name == 'Mobilenetv3_small':
        model = mobilenet_v3_small(weights=weights)
        assert model.__class__.__name__ == 'MobileNetV3'
        model.classifier[3] = nn.Linear(in_features=1024,out_features=num_classes)
    return model

# Cross entropy and evidential test code.
def test_one_epoch(dataloader,model,device,loss_function):
    print("Started testing")    
    
    images_list = []
    labels_list = []
    image_paths = []

    true_labels = []
    predicted_labels = []

    softmax_probabilities = []
    softmax_confidences = []
    cross_entropy_output = []

    evidential_probabilities = []
    dirichlet_alpha_output = []
    evidential_confidences = []

    # Begin testing
    with torch.no_grad():
        for batch_idx,(inputs,labels,images_paths) in enumerate(dataloader):
            inputs,labels = inputs.to(device),labels.to(device)
            
            for image,label,img_path in zip(inputs,labels,images_paths):
                images_list.append(image)
                labels_list.append(label.cpu().numpy)
                image_paths.append(img_path)
            # CROSS ENTROPY
            if loss_function == 'Crossentropy':
                print("Testing for cross entropy loss")  
                model.eval()
                model.to(device=device)
                # Get the logits
                logits = model(inputs)
                # Save the logits
                cross_entropy_output.extend(logits.cpu().numpy())
                # Softmax output
                outputs = torch.softmax(logits,dim=1)
                # Save softmax output
                softmax_probabilities.extend(outputs.cpu().numpy())
                # Get the confidences
                confidences,predictions = torch.max(outputs,1)
                # Save confidences
                softmax_confidences.extend(confidences.cpu().numpy())
                # Save predicted labels.
                predicted_labels.extend(predictions.cpu().numpy())
                # Save true labels
                true_labels.extend(labels.cpu().numpy())
            # EVIDENTIAL
            elif loss_function == 'Evidential':
                print("Testing for evidential loss")  
                model.eval()
                model.to(device=device)
                # Get the logits
                logits = model(inputs)
                # print("logits",logits.shape)
                # Save alpha values for evidential entropy calculation
                evidence = relu_evidence(logits)
                alpha = evidence + 1
                # print("alpha",alpha.shape)
                # Save the dirchlet logits
                dirichlet_alpha_output.extend(alpha.cpu().numpy())
                # print("alpha outputs",len(dirichlet_alpha_output))
                # # Save true labels
                true_labels.extend(labels.cpu().numpy())


    # Save cross entropy results
    if loss_function == 'Crossentropy':
        # Convert the results to numpy arrays
        softmax_probs_array = [i/np.sum(i) for i in softmax_probabilities]
        pred_labels_array = np.array(predicted_labels)
        true_labels_array = np.array(true_labels)
        cross_entropy_logits_array = np.array(cross_entropy_output)
        softmax_confidences_array = np.array(softmax_confidences)

        ce_results_dict = {
                        "true_labels":true_labels_array,
                        "pred_labels":pred_labels_array,
                        "probabilities":np.array(softmax_probs_array),
                        "model_output":cross_entropy_logits_array,
                        "images_list": images_list,
                        "confidences": softmax_confidences_array,
                        "image_paths": image_paths
        }
        return ce_results_dict
        
    # Save evidential results
    elif loss_function == 'Evidential':
        
        evidential_probabilities = [i/np.sum(i) for i in dirichlet_alpha_output]
        evidential_probs_array = np.array(evidential_probabilities)
        evidential_confidences =np.max(evidential_probs_array,axis=1)
        predicted_labels =np.argmax(evidential_probs_array,axis=1)
        # Convert the results to numpy arrays
        evidential_logits_array = np.array(dirichlet_alpha_output)
        evidential_confidences_array = np.array(evidential_confidences)
        pred_labels_array = np.array(predicted_labels)
        true_labels_array = np.array(true_labels)
        
        evi_results_dict = {
                        "true_labels":true_labels_array,
                        "pred_labels":pred_labels_array,
                        "probabilities":evidential_probs_array,
                        "model_output":evidential_logits_array,
                        "images_list": images_list,
                        "confidences": evidential_confidences_array,
                        "image_paths": image_paths
        }
        return evi_results_dict
    
####################### MOHAN############################
# Cross entropy and evidential test code.
def test_epoch_1(dataloader,model,device,loss_function):
    print("Started testing")    
    
    images_list = []
    labels_list = []
    image_paths = []

    true_labels = []
    predicted_labels = []

    softmax_probabilities = []
    softmax_confidences = []
    cross_entropy_output = []

    evidential_probabilities = []
    dirichlet_alpha_output = []
    evidential_confidences = []

    # Begin testing
    with torch.no_grad():
        for batch_idx,(inputs,labels,file_paths) in enumerate(dataloader):
            inputs,labels = inputs.to(device),labels.to(device)
            
            for image,label in zip(inputs,labels):
                images_list.append(image)
                labels_list.append(label.cpu().numpy)
            # CROSS ENTROPY
            if loss_function == 'Crossentropy':
                print("Testing for cross entropy loss")  
                model.eval()
                model.to(device=device)
                # Get the logits
                logits = model(inputs)
                # Save the logits
                cross_entropy_output.extend(logits.cpu().numpy())
                # Softmax output
                outputs = torch.softmax(logits,dim=1)
                # Save softmax output
                softmax_probabilities.extend(outputs.cpu().numpy())
                # Get the confidences
                confidences,predictions = torch.max(outputs,1)
                # Save confidences
                softmax_confidences.extend(confidences.cpu().numpy())
                # Save predicted labels.
                predicted_labels.extend(predictions.cpu().numpy())
                # Save true labels
                true_labels.extend(labels.cpu().numpy())
            # EVIDENTIAL
            elif loss_function == 'Evidential':
                print("Testing for evidential loss")  
                model.eval()
                model.to(device=device)
                # Get the logits
                logits = model(inputs)
                # print("logits",logits.shape)
                # Save alpha values for evidential entropy calculation
                evidence = relu_evidence(logits)
                alpha = evidence + 1
                # print("alpha",alpha.shape)
                # Save the dirchlet logits
                dirichlet_alpha_output.extend(alpha.cpu().numpy())
                # print("alpha outputs",len(dirichlet_alpha_output))
                # # Save true labels
                true_labels.extend(labels.cpu().numpy())


    # Save cross entropy results
    if loss_function == 'Crossentropy':
        # Convert the results to numpy arrays
        softmax_probs_array = [i/np.sum(i) for i in softmax_probabilities]
        pred_labels_array = np.array(predicted_labels)
        true_labels_array = np.array(true_labels)
        cross_entropy_logits_array = np.array(cross_entropy_output)
        softmax_confidences_array = np.array(softmax_confidences)

        ce_results_dict = {
                        "true_labels":true_labels_array,
                        "pred_labels":pred_labels_array,
                        "probabilities":np.array(softmax_probs_array),
                        "model_output":cross_entropy_logits_array,
                        "images_list": images_list,
                        "confidences": softmax_confidences_array,
                        "image_paths": image_paths
        }
        return ce_results_dict
        
    # Save evidential results
    elif loss_function == 'Evidential':
        
        evidential_probabilities = [i/np.sum(i) for i in dirichlet_alpha_output]
        evidential_probs_array = np.array(evidential_probabilities)
        evidential_confidences =np.max(evidential_probs_array,axis=1)
        predicted_labels =np.argmax(evidential_probs_array,axis=1)
        # Convert the results to numpy arrays
        evidential_logits_array = np.array(dirichlet_alpha_output)
        evidential_confidences_array = np.array(evidential_confidences)
        pred_labels_array = np.array(predicted_labels)
        true_labels_array = np.array(true_labels)
        
        evi_results_dict = {
                        "true_labels":true_labels_array,
                        "pred_labels":pred_labels_array,
                        "probabilities":evidential_probs_array,
                        "model_output":evidential_logits_array,
                        "images_list": images_list,
                        "confidences": evidential_confidences_array,
                        "image_paths": image_paths
        }
        return evi_results_dict


####################################################################


def get_best_worst_predictions(confidence_array):
    confidences_idxs_sorted = np.argsort(confidence_array)
    best_predictions = confidences_idxs_sorted[-9 : ]
    worst_predictions = confidences_idxs_sorted[:9]

    return best_predictions,worst_predictions

def plot_predicted_images(predictions,confidences_pred_images,image_paths,labels,class_names,plot_preds,results_path,plot_name):

    rows = 3
    columns = 3
    fig = plt.figure(figsize=(20,20))
    
    for idx,i in enumerate(plot_preds):
        idx = idx+1
        image = image_paths[i]
        img_ = mpimg.imread(str(image))
        # img_ = images[i].cpu().detach()
        label_true = labels[i]
        label_pred = predictions[i]
        confidence = confidences_pred_images [i]

        fig.add_subplot(rows, columns, idx)
        # showing image
        plt.imshow(img_)
        plt.title("True_label : "+ str(class_names[label_true]))
        plt.xlabel("Confidance : " + str(round(confidence,2)))
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
    y = torch.eye(num_classes).cuda()
    return y[labels]

def save_architecture_txt(model,dir_path,filename):
        complete_file_name = os.path.join(dir_path, filename+"_arch.txt")
        with open(complete_file_name, "w") as f:
                f.write(str(model))
                f.close()
        return None

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def get_accuracy_score(true_labels,predicted_labels):
  return accuracy_score(true_labels,predicted_labels)

def get_precision_score(true_labels,predicted_labels):
  return precision_score(true_labels,predicted_labels,average='weighted')

def plot_confusion_matrix1(true_labels,predicted_labels,class_names,results_path,plot_name):
        fig, axs = plt.subplots(figsize=(25, 25))
        plot_confusion_matrix(true_labels,predicted_labels , ax=axs,normalize=True)
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=90, fontsize=13)
        plt.yticks(tick_marks, class_names, rotation=0,fontsize=13)
        plt.title('Confusion Matrix', fontsize=18)
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

def plot_losses(train_loss,valid_loss,criterion_name,save_path):
        fig, ax = plt.subplots(figsize=(20,20))
        ax.plot(train_loss,label='Training Loss')
        ax.plot(valid_loss,label='Validation Loss')
        plt.xlabel('# Epoch', fontsize=15)
        plt.ylabel(str(criterion_name),fontsize=15)
        plt.title('Loss Plot')
        plt.legend()
        fig.savefig(str(save_path)+'/losses_plot.png')
        return fig

def plot_accuracies(train_acc,valid_acc,save_path):
        fig, ax = plt.subplots(figsize=(20,20))
        ax.plot(train_acc,label='Training accuracy')
        ax.plot(valid_acc,label='Validation accuracy')
        plt.xlabel('# Epoch')
        plt.ylabel('Accuracy')
        plt.title("Accuracy Plot")
        plt.legend()
        fig.savefig(str(save_path)+'/accuracies_plot.png')
        return fig

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
