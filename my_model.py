import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms,models
# import matplotlib as plt
import matplotlib.pyplot as plt 
import os
import time

batch_size = 500

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cuda')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

training_datasets = datasets.CIFAR10(root='.',download=True ,transform= transform,train=True)
testing_datasets = datasets.CIFAR10(root='.',download=True ,transform= transform,train=False)


training_loader = torch.utils.data.DataLoader(training_datasets, batch_size=batch_size, shuffle=True,num_workers=12)
testning_loader = torch.utils.data.DataLoader(testing_datasets, batch_size=batch_size, shuffle=False,num_workers=12)


classes = ('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')

class omar_Convolutional_network(nn.Module):
    def __init__(self):
        super(omar_Convolutional_network,self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(3, 6 , 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16 , 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class omar_MLP(nn.Module):
    def __init__(self):
        super(omar_MLP,self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32*32*3, 900)
        self.fc2 = nn.Linear(900, 600)
        self.fc3 = nn.Linear(600, 400)
        self.fc4 = nn.Linear(400, 300)
        self.fc5 = nn.Linear(300, 10)

    
    def forward(self,x):
        x = x.view(-1, 32*32*3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        return x

class SevenLayerFC_Net(nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate three nn.Linear modules and assign them as
        member variables.
        """
        super(SevenLayerFC_Net, self).__init__()
        N, D_in, H, D_out = batch_size, 3072, 200, 10
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, D_out)

        
    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        x = x.view(-1, 32*32*3)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x  
    

model = models.densenet121(pretrained=False)

# model = omar_Convolutional_network()
#to freeze our feature parameters
# for param in model.parameters():
#     param.requires_grad = False

# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs,10)
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
    ('fc1',nn.Linear(1024,500)),
    ('relu',nn.ReLU()),
    ('fc2',nn.Linear(500,10)),
    ('output',nn.LogSoftmax(dim=1))
]))
model.classifier = classifier

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
list_of_training_acc=[]
list_of_val_acc=[]
list_of_epochs=[]
highest_accuracy=0
epoch_of_highest_accuracy=0

first_time = time.time()
n_total_steps = len(training_loader)


for epoch in range(num_epochs):
    i=0
    training_loss=0
    val_loss=0

    
    training_correct = 0
    training_samples = 0

    testing_correct = 0
    testing_samples = 0
    
    for images,labels in training_loader:
        images = images.to(device)
        labels = labels.to(device)
        model.train()

        outputs = model(images)

        x,prediction =torch.max(outputs, 1)
        training_samples += labels.size(0)
        
        # break
        training_correct += (prediction == labels).sum().item()
        training_loss = criterion(outputs, labels)

        optimizer.zero_grad()
        training_loss.backward()
        optimizer.step()

        i+=1
    with torch.no_grad():
        for images,labels in testning_loader:
            images = images.to(device)
            labels = labels.to(device)

            model.eval()
            outputs = model(images)
            
            x,prediction =torch.max(outputs, 1)
            testing_samples += labels.size(0)
            testing_correct += (prediction == labels).sum().item()

            val_loss = criterion(outputs, labels)

            # optimizer.zero_grad()
            # val_loss.backward()
            # optimizer.step()

            i+=1
    
    training_accuracy= 100* training_correct/ training_samples
    list_of_training_acc.append(training_accuracy)
    testing_accuracy= 100* testing_correct/ testing_samples
    list_of_val_acc.append(testing_accuracy)
    list_of_epochs.append(epoch+1)

    if testing_accuracy > highest_accuracy:
        highest_accuracy = testing_accuracy
        epoch_of_highest_accuracy=epoch
        

    print("")
    print(epoch,f' For Training Loss: {training_loss.item():.4f}'," accuracy: ",training_accuracy,'%') 
    print(epoch,f' For validating Loss: {val_loss.item():.4f}'," accuracy: ",testing_accuracy,"%")
    print('time of this epoch: ',int(time.time()-first_time))
# plt.plot([list_of_training_acc], list_of_epochs)
# plt.plot(list_of_val_acc, list_of_epochs)
# plt.xlabel("Time (s)")
# plt.ylabel("Scale (Bananas)")
print("======================================================================================")
print('highest accuraty until now is ',highest_accuracy," at",epoch_of_highest_accuracy)
print("======================================================================================")
plt.plot(list_of_epochs,list_of_training_acc,label='training')
plt.plot(list_of_epochs,list_of_val_acc,label='validation')
plt.xlabel("Epoch")
plt.ylabel("accuracy")
plt.legend()

plt.show()
