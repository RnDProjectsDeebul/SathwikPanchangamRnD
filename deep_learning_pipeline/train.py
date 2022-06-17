# Import modules
import torch


# Training class
class Training():
    def __init__(self,num_epochs:int,model = None,criterion = None,optimizer = None,trainloader = None,testloader = None,logger = None):
        
        self.num_epochs = num_epochs
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.testloader = testloader
        self.run = logger

        self.train_losses = []
        self.test_losses = []
        self.y_true = []
        self.y_pred = []

    def train(self,device):
        total_train_loss = 0.0
        total_test_loss = 0.0
        for epoch in range(self.num_epochs):
            total_train_loss = 0
            for images , labels in self.trainloader:
                images = images.to(device)
                labels = labels.to(device)
                self.optimizer.zero_grad()
                logits = self.model(images)
                _,preds = torch.max(logits,1)
                loss = self.criterion(logits, labels)
                acc = (torch.sum(preds == labels).item())/len(labels)
                
                self.run['train/epoch/train_loss'].log(loss)
                self.run['train/epoch/train_accuracy'].log(acc)

                total_train_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            else:
                total_train_loss = 0
                num_correct_predictions = 0

                with torch.no_grad():
                    for images, labels in self.testloader:
                        images = images.to(device)
                        labels = labels.to(device)
                        logits = self.model(images)
                        _,preds = torch.max(logits,1)

                        acc = (torch.sum(preds == labels).item())/len(labels)
                
                        output = (torch.max(torch.exp(logits), 1)[1]).data.cpu().numpy()
                        self.y_pred.extend(output)
                        self.y_true.extend(labels)
                        loss = self.criterion(logits, labels)

                        self.run['train/epoch/test_loss'].log(loss)
                        self.run['train/epoch/test_accuracy'].log(acc)
                        total_test_loss += loss.item()

                        probabilities = torch.exp(logits)
                        top_p, top_class = probabilities.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        num_correct_predictions += equals.sum().item()
                 # Calculating the mean loss to enable comparison between training and test sets.
                train_loss = total_train_loss/len(self.trainloader.dataset)
                test_loss = total_test_loss/len(self.testloader.dataset)

                # Creating list of train and test losses
                self.train_losses.append(train_loss)
                self.test_losses.append(test_loss)


                # Printing the results
                print("Epoch: {}/{}... ".format(epoch+1,self.num_epochs),
                "Training Loss: {:.3f}... ".format(train_loss),
                "Test Loss: {: .3f}... ".format(test_loss),
                "Test Accuracy: {: .3f}".format(num_correct_predictions/len(self.testloader.dataset)))
                
        return [self.train_losses,self.test_losses, self.y_true,self.y_pred]