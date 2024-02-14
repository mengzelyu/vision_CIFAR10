import torch
import torchvision
import torchvision.transforms as transforms
import pathlib
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(3, 9, 3, padding=1)
        # self.pool1 = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(9, 27, 3, padding=1)
        # self.pool2 = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(32 * 32 * 3, 512)
        # self.fc2 = nn.Linear(512, 128)
        # self.fc3 = nn.Linear(128, 32)
        # self.fc4 = nn.Linear(32, 10)

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)  # Input: 32x32x3, Output: 32x32x32
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)  # Input: 32x32x32, Output: 32x32x64
        self.pool1 = nn.MaxPool2d(2, 2)  # Input: 32x32x64, Output: 16x16x64
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # Input: 16x16x64, Output: 16x16x128
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)  # Input: 16x16x128, Output: 16x16x128
        self.pool2 = nn.MaxPool2d(2, 2)  # Input: 16x16x128, Output: 8x8x128
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)  # Input: 8x8x128, Output: 8x8x256
        self.pool3 = nn.MaxPool2d(2, 2)  # Input: 8x8x256, Output: 4x4x256

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

        # Fully connected layers
        self.fc1 = nn.Linear(4 * 4 * 256, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)  # Assuming 10 classes

    def forward(self, x):
        # x = self.pool1(F.relu(self.conv1(x)))
        # x = self.pool2(F.relu(self.conv2(x)))
        # x = x.view(-1, 32 * 32 * 3)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = self.fc4(x)

        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = self.pool3(F.relu(self.conv5(x)))

        # Flatten the output for the fully connected layers
        x = x.view(-1, 4 * 4 * 256)

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    pathlib.Path().resolve()
    print("CUDA available: ", torch.cuda.is_available())
    print("Device: ", torch.cuda.get_device_name())

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.25)

    losses = []

    for epoch in range(7):  # loop over the dataset multiple times
        print(f"lr = {optimizer.param_groups[0]['lr']}")

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1000 == 999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                losses.append(running_loss / 1000)
                running_loss = 0.0
        scheduler.step()
    plt.plot(losses)
    plt.show()
        # if input("Early Stop?") == "Y":
        #     break
    print('Finished Training')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))

