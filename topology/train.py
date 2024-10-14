import torch
import torchvision
import torchvision.transforms as transforms
from topology.topologies import *

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

start_lr = 0.99
effective_batch_size = 16
num_workers = 8
batch_size = num_workers * effective_batch_size
epochs = 10
topology = "Random exponential"

PATH = './essai1_' + topology + str(epochs) +'epoch' + str(start_lr) + 'learning_rate.pth'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


working_CNNs = [Net().to(device) for i in range(num_workers)]

train_losses = [[]]*num_workers
test_acuracy = []

criterion = nn.CrossEntropyLoss()
optimizers = [optim.SGD(worker.parameters(), lr=start_lr, momentum=0.0) for worker in working_CNNs]

for epoch in range(epochs):  # loop over the dataset multiple times
    
    running_losses = [0.0]*num_workers
    for i, big_batch in enumerate(trainloader, 0):
        #for each time step, we conduct one computation (gradient descent) step, followed by one communication step
        
        #computation step
        big_inputs, big_labels = big_batch
        for k in range(num_workers):
            # get the inputs; data is a list of [inputs, labels]
            inputs = big_inputs[effective_batch_size*k : effective_batch_size*(k+1)].to(device)
            labels = big_labels[effective_batch_size*k : effective_batch_size*(k+1)].to(device)
            

            # zero the parameter gradients
            optimizers[k].zero_grad()

            # forward + backward + optimize
            outputs = working_CNNs[k](inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizers[k].step()

            # print statistics
            running_losses[k] += loss.item()
            train_losses[k].append(loss.item())

        if i % 60 == 59:    # print every 2000 mini-batches
            print(f'[epoch : {epoch + 1}, average running loss accross workers, iteration : {i + 1:5d}] loss: {sum(running_losses) / (num_workers*60) :.3f}')
            running_losses = [0.0]*num_workers  
            

        # communication step
        W = scheme_for_string(topology, num_workers=num_workers).w(i)

        with torch.no_grad():
            for param_name, param in working_CNNs[0].named_parameters():
                new_param = [param.clone().zero_() for _ in range(num_workers)]
                for i in range(num_workers):
                    for j in range(num_workers):
                        new_param[i] += W[i,j]*(working_CNNs[j].state_dict()[param_name])
                for k in range(num_workers):
                    working_CNNs[k].state_dict()[param_name].data.copy_(new_param[k])

    # Accuracy test on test dataset
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = working_CNNs[0](images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct // total
    test_acuracy.append(accuracy)
    print(f'Accuracy at epoch {epoch} on the 10000 test images: {100 * correct // total} %')

print('Finished Training')

with open('train_losses.txt', 'w') as file:
    for loss_one_worker in train_losses:
        file.write(','.join(map(str, loss_one_worker)) + '\n')

with open('test_acuracy.txt', 'w') as file:
    file.write(','.join(map(str, test_acuracy)) + '\n')

torch.save(working_CNNs[0].state_dict(), PATH)