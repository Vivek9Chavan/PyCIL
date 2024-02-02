import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])

#trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                      #download=True, transform=transform)

traindir = '/mnt/1TBNVME/vivek/2024/MNIST_JPG/train/'
valdir = '/mnt/1TBNVME/vivek/2024/MNIST_JPG/val/'

trainset = torchvision.datasets.ImageFolder(traindir, transform=transform)
trainloader = DataLoader(trainset, batch_size=512, shuffle=True)

#testset = torchvision.datasets.MNIST(root='./data', train=False,
                                     #download=True, transform=transform)
testset = torchvision.datasets.ImageFolder(valdir, transform=transform)
testloader = DataLoader(testset, batch_size=512, shuffle=False)

from time import time


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28*3, 256)  # First fully connected layer
        self.fc2 = nn.Linear(256, 512)     # Second fully connected layer
        self.fc3 = nn.Linear(512, 1024)      # Third fully connected layer
        self.fc4 = nn.Linear(1024, 512)      # Fourth fully connected layer
        self.fc5 = nn.Linear(512, 256)      # Fifth fully connected layer
        self.fc6 = nn.Linear(256, 10)      # Output layer

    def forward(self, x):
        x = x.view(-1, 28*28*3)  # Flatten the image
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = torch.log_softmax(self.fc6(x), dim=1)
        return x

model = MLP()

time1 = time()

model = MLP().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


for epoch in range(10):  # loop over the dataset multiple times
    print(f'Epoch {epoch + 1}')

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)


        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    #print accuracy and loss after each epoch
    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()


    print(f'Accuracy of the network on the train images: {100 * correct // total} %')


    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')
    print(f'Loss: {loss.item()}')

time2 = time()

print('Finished Training in {} seconds'.format(time2-time1))


# save the model
torch.save(model.state_dict(), 'mlp_model_29_01.pth')

from PIL import Image

test_image = '/mnt/1TBNVME/vivek/2024/MNIST_JPG/9_test.jpg'
test_image = Image.open(test_image)
#convert to 28*28
test_image = test_image.resize((28,28))
test_image = transform(test_image)
test_image = test_image.unsqueeze(0)
test_image = test_image.to(device)

model.eval()
with torch.no_grad():
    output = model(test_image)
    print(output)
    _, predicted = torch.max(output.data, 1)
    print(predicted.item())