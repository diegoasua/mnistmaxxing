import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

transform1 = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.RandomApply(
            [transforms.RandomAffine(degrees=10, translate=(0.1, 0.1))], p=0.5
        ),
    ]
)

trainset = datasets.MNIST(
    "~/.pytorch/MNIST_data/", download=True, train=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=32, shuffle=True, num_workers=6
)

testset = datasets.MNIST(
    "~/.pytorch/MNIST_data/", download=True, train=False, transform=transform1
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=32, shuffle=True, num_workers=6
)


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.fc1 = nn.Linear(32 * 1 * 1, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm1d(120)
        self.bn5 = nn.BatchNorm1d(84)

    def forward(self, x):
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = nn.MaxPool2d(2)(x)
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        x = nn.MaxPool2d(2)(x)
        x = nn.ReLU()(self.bn3(self.conv3(x)))
        x = nn.MaxPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.bn4(self.fc1(x)))
        x = nn.ReLU()(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        return x


net = LeNet5()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

if __name__ == "__main__":
    net.to(device)
    criterion.to(device)

    num_epochs = 30

    start_time = time.time()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        scheduler.step()
        print("Epoch %d, Loss: %.3f" % (epoch + 1, running_loss / (i + 1)))

    end_time = time.time()
    training_time = end_time - start_time

    print("Training completed in %.2f seconds" % training_time)

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(
        "Accuracy of the network on the 10000 test images: %.2f %%"
        % (100 * correct / total)
    )
