import torch
import time
from helpers import get_device, get_accuracy_score,get_classification_report,get_f1_score,get_precision_score,get_recall_score, one_hot_embedding
from torch.autograd import Variable
from torchmetrics import Accuracy
from losses import relu_evidence


def train_model(model=None,
                num_epochs=None,
                uncertainty = None,
                criterion=None,
                optimizer=None,
                scheduler=None,
                dataloaders=None,
                class_names=None,
                logger=None):
    since = time.time()

    num_classes = len(class_names)

    train_losses = []
    valid_losses = []

    train_trues = []
    train_preds = []

    valid_trues = []
    valid_preds = []

    # Checking the device
    device = get_device()
    print("Using device :", device)

    # Set model to gpu
    if device:
        model.to(device)

    # train the model
    best_model_wts = model.state_dict()
    best_acc = 0.0

    # Accuracy pytorch metrics
    train_accuracy = Accuracy(num_classes = num_classes).to(device)
    valid_accuracy = Accuracy(num_classes = num_classes).to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
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
                if device:
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
                    alpha = evidence + 1
                    u = num_classes / torch.sum(alpha, dim=1, keepdim=True)

                    total_evidence = torch.sum(evidence, 1, keepdim=True)
                    mean_evidence = torch.mean(total_evidence)
                    mean_evidence_succ = torch.sum(
                        torch.sum(evidence, 1, keepdim=True) * match) / torch.sum(match + 1e-20)
                    mean_evidence_fail = torch.sum(
                        torch.sum(evidence, 1, keepdim=True) * (1 - match)) / (torch.sum(torch.abs(1 - match)) + 1e-20)

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
                    # train_losses.append(running_loss)
                    #train_trues.append(labels.cpu())
                    #train_preds.append(preds.cpu())
                    train_accuracy.update(preds,labels)
                else:
                    # valid_losses.append(running_loss)
                    #valid_trues.append(labels.cpu())
                    #valid_preds.append(preds.cpu())
                    valid_accuracy.update(preds,labels)

            epoch_loss = running_loss / len(dataloaders[phase])
            logger['plots/loss/'+str(phase)].log(epoch_loss)

            if phase == 'train':
                epoch_acc_train = train_accuracy.compute()
                train_accuracy.reset()
                logger['plots/accuracy/train_accuracy'].log(epoch_acc_train)
            else:
                epoch_acc_valid = valid_accuracy.compute()
                valid_accuracy.reset()
                logger['plots/accuracy/valid_accuracy'].log(epoch_acc_valid)
            # epoch_acc = running_corrects / len(dataloaders[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc_train if phase == 'train' else epoch_acc_valid ))

            # deep copy the model
            if phase == 'val' and epoch_acc_valid > best_acc:
                best_acc = epoch_acc_valid
                best_model_wts = model.state_dict()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,train_losses,valid_losses



# TODO: Add early stopping, check if you need to return anything else. also plot confusion matrix, losses, etc using seaborn after saving the values as pandas dataframes(csv files)