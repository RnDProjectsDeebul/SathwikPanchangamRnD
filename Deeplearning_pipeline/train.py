import torch
from helpers import get_device, get_accuracy_score,get_classification_report,get_f1_score,get_precision_score,get_recall_score

def train_model(model=None,num_epochs=None,criterion=None,optimizer=None,scheduler=None,dataloader=None,class_names=None,logger=None):

    model.train()

    train_losses = []
    test_losses = []

    y_trues = []
    y_preds = []
    # Checking the device
    device = get_device()
    print("Using device :", device)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(dataloader,0):
            model.to(device)
            # images, labels = data
            images , labels = data[0].to(device), data[1].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i%100==99:
                print(f'[{epoch + i}, {i + 1:5d}] loss:{running_loss/100:.3f}')
                running_loss=0.0
    path = '/home/sathwikpanchngam/rnd/github_projects/SathwikPanchangamRnD/Deeplearning_pipeline/results'
    torch.save(model.state_dict(),path)
    print('Finished training')