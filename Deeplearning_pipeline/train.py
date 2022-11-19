import torch
import time
from helpers import early_stopping, get_device, get_accuracy_score,get_classification_report,get_f1_score,get_precision_score,get_recall_score, one_hot_embedding
from torch.autograd import Variable
from torchmetrics import Accuracy
from losses import relu_evidence
import pandas as pd
import numpy as np

def train_model(model=None,
                num_epochs=None,
                uncertainty = None,
                criterion=None,
                optimizer=None,
                scheduler=None,
                dataloaders=None,
                class_names=None,
                logger=None,
                results_file_path = None):
    num_classes = len(class_names)

    losses_accu_dict = {"train":[],"val":[],"train_acc":[],"valid_acc":[]}
    training_results_dict = {"train_trues":None,"train_preds":None}
    
    validation_results_dict = {"valid_trues":None,"valid_preds":None}

    train_trues = []
    train_preds = []
    valid_trues = []
    valid_preds = []
    probabilities_list_train = []
    probabilities_list_valid = []
    # Checking the device
    device = get_device()
    print("Using device :", device)

    # Set model to gpu
    if torch.cuda.is_available():
        model.to(device)

    # train the model
    best_model_wts = model.state_dict()
    best_acc = 0.0

    # Accuracy pytorch metrics
    train_accuracy = Accuracy(num_classes = num_classes).to(device)
    valid_accuracy = Accuracy(num_classes = num_classes).to(device)

    # Time 
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                if logger !=None:
                    logger['plots/lr/learning_rate'].log(scheduler.get_lr())
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)

                if uncertainty:
                    y = one_hot_embedding(labels=labels,num_classes=num_classes)
                    y.to(device=device)
                    loss = criterion(
                            outputs, y.float(), epoch, num_classes, 10,device)
                    
                    match = torch.reshape(torch.eq(
                            preds, labels).float(), (-1, 1))
                    acc = torch.mean(match)

                    evidence = relu_evidence(outputs)
                    # print(evidence)   ## deebul 
                    alpha = evidence + 1
                    u = num_classes / torch.sum(alpha, dim=1, keepdim=True)

                    total_evidence = torch.sum(evidence, 1, keepdim=True)
                    mean_evidence = torch.mean(total_evidence)
                    mean_evidence_succ = torch.sum(
                        torch.sum(evidence, 1, keepdim=True) * match) / torch.sum(match + 1e-20)
                    mean_evidence_fail = torch.sum(
                        torch.sum(evidence, 1, keepdim=True) * (1 - match)) / (torch.sum(torch.abs(1 - match)) + 1e-20)

                    # outputs = [i/torch.sum(outputs) for i in outputs]

                else:
                    loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                # statistics
                running_loss += loss.item()
                # running_corrects += torch.sum(preds == labels.data).to(torch.float32)
                

                # Store the results
                if phase == 'train':
                    # for label,prediction in zip(labels,preds):
                    #     train_trues.append(label)
                    #     train_preds.append(prediction)
                    train_trues.extend(labels.cpu().detach().numpy())
                    train_preds.extend(preds.cpu().detach().numpy())
                    # probabilities_list_train.extend(outputs)
                    train_accuracy.update(preds,labels)
                else:
                    # for label,prediction in zip(labels,preds):
                    #     valid_trues.append(label)
                    #     valid_preds.append(prediction)
                    valid_trues.extend(labels.cpu().detach().numpy())
                    valid_preds.extend(preds.cpu().detach().numpy())
                    # probabilities_list_valid.extend(outputs)
                    valid_accuracy.update(preds,labels)

            epoch_loss = running_loss / len(dataloaders[phase])
            if logger !=None:
                logger['plots/loss/'+str(phase)].log(epoch_loss)

            losses_accu_dict[phase].append(epoch_loss) #spanch2s

            if phase == 'train':
                epoch_acc_train = train_accuracy.compute()
                losses_accu_dict['train_acc'] = epoch_acc_train.cpu().detach().numpy()
                if logger != None:
                    logger['plots/accuracy/train_accuracy'].log(epoch_acc_train)
                train_accuracy.reset()
            else:
                epoch_acc_valid = valid_accuracy.compute()
                losses_accu_dict['valid_acc'] = epoch_acc_valid.cpu().detach().numpy()
                if logger !=None:
                    logger['plots/accuracy/valid_accuracy'].log(epoch_acc_valid)
                valid_accuracy.reset()
            # epoch_acc = running_corrects / len(dataloaders[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc_train*100 if phase == 'train' else epoch_acc_valid*100 ))

            # Store the training results
            training_results_dict['train_trues'] = train_trues
            training_results_dict['train_preds'] = train_preds
            # training_results_dict['train_probs'] = probabilities_list_train
            validation_results_dict['valid_trues'] = valid_trues
            validation_results_dict['valid_preds'] = valid_preds
            # validation_results_dict['valid_probs'] = probabilities_list_valid

            # deep copy the model
            if phase == 'val' and epoch_acc_valid > best_acc:
                best_acc = epoch_acc_valid
                best_model_wts = model.state_dict()

            # # You may need to change your approach for loss calculation for phases to add early stopping
            # if early_stopping(epoch_train_loss,epoch_validation_loss,min_delta=10,tolerance=20):
            #     print("Current epoch is : ", epoch)
            #     break


    print('Best val Acc: {:4f}'.format(best_acc))

    # End time
    end.record()
    torch.cuda.synchronize()

    print('Finished Training',start.elapsed_time(end)/600)

    df = pd.DataFrame(losses_accu_dict,columns=["train","val","train_acc","valid_acc"])
    df1 = pd.DataFrame(training_results_dict,columns=["train_trues","train_preds"])
    df2 = pd.DataFrame(validation_results_dict,columns=["valid_trues","valid_preds"])

    df.to_csv(path_or_buf=str(results_file_path)+'/losses_accuracy_original.csv')
    df1.to_csv(path_or_buf=str(results_file_path + '/train_results_original.csv'))
    df2.to_csv(path_or_buf=str(results_file_path)+'/valid_results_original.csv')
    
    # save the best model weights
    model.load_state_dict(best_model_wts)
    return model



# TODO: Add early stopping, check if you need to return anything else. also plot confusion matrix, losses, etc using seaborn after saving the values as pandas dataframes(csv files)