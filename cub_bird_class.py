import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
import sys

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_classes = 10

# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def setup_database():
    # Set up dataset and classes
    transform_train = transforms.Compose(
        [#transforms.Resize((224,224), interpolation=Image.NEAREST),
         #transforms.Resize((32,32), interpolation=Image.NEAREST),
         #transforms.Resize((84,84), interpolation=Image.NEAREST),

         #transforms.RandomHorizontalFlip(),
         #transforms.RandomRotation(15),
         transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,)),
         #transforms.Normalize((0.485, 0.485, 0.485), (0.229, 0.224, 0.225))
         ])


    transform_val_test = transforms.Compose(
         [transforms.Resize((400,400), interpolation=Image.NEAREST),
          transforms.ToTensor(),
          #transforms.Normalize((0.485, 0.485, 0.485), (0.229, 0.224, 0.225))
          ])

    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize((0.5,), (0.5,)),
                                  ])

    # Download and load the training data
    trainSet = torchvision.datasets.MNIST('./data', download=True, train=True, transform=transform)
    valset = torchvision.datasets.MNIST('./data', download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainSet, batch_size=128, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=True, num_workers=2)
    testloader = valloader

    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    print(len(trainSet))
    print(len(valset))
    print('trainloader length: ', len(trainloader.dataset))
    print('valloader length: ', len(valloader.dataset))
    print('testloader length: ', len(testloader.dataset))

    return (trainSet, trainloader, valloader, testloader, classes)

# same uncertainty measure as seen in that medium post
# uncertainty = most confident - least confident
# we want to put the smallest uncertainties in our training
# because that means that that sample is not well defined

def largest_margin(batch):
    max_batch = 0
    min_batch = 0
    for out in batch:
        max_batch += max(out)
        min_batch += min(out)

    confidence = max_batch - min_batch

    return confidence

def smallest_margin(batch):
    max_batch = 0
    max_batch_sec = 0
    for out in batch:
        out1 = np.sort(out)
        max_batch += out1[-1]
        max_batch_sec += out1[-2]

    confidence = max_batch - max_batch_sec

    return confidence


def get_active_batches(trainloader, net, sampling, num_batches=1, print_f=False):
    act_dict = {}
    count_key = 0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = [d.to(device) for d in data]

        outputs = net(inputs)

        outputs_cpu = outputs.cpu()
        outputs_np = outputs_cpu.detach().numpy()

        if sampling == 'smallest_margin':
            confidence = smallest_margin(outputs_np)
            if len(act_dict.keys()) < num_batches:
                act_dict[count_key] = [confidence, inputs, labels]
                count_key += 1
            else:
                max_idx = list(act_dict.keys()).index(max(list(act_dict.keys())))
                max_key = list(act_dict.keys())[max_idx]

                if act_dict[max_key][0] > confidence:
                    act_dict[max_key] = [confidence, inputs, labels]
        elif sampling == 'random':
            if len(act_dict.keys()) < num_batches:
                act_dict[count_key] = [i, inputs, labels]
                count_key += 1
        elif sampling == 'largest_margin':
            confidence = largest_margin(outputs_np)
            if len(act_dict.keys()) < num_batches:
                act_dict[count_key] = [confidence, inputs, labels]
                count_key += 1
            else:
                max_idx = list(act_dict.keys()).index(max(list(act_dict.keys())))
                max_key = list(act_dict.keys())[max_idx]

                if act_dict[max_key][0] > confidence:
                    act_dict[max_key] = [confidence, inputs, labels]
        else:
            print('Sampling Type in get_active_batches not given')
            sys.exit(0)


    if print_f:
        print('smallest confidences:', [act_dict[key][0] for key in act_dict])
    active_data = []
    active_labels = []
    for key in act_dict:
        active_data.append(act_dict[key][1])
        active_labels.append(act_dict[key][2])

    return active_data, active_labels

def train(net, inputs, labels, print_f=False):
    assert len(inputs) == len(labels)
    for i in range(len(inputs)):
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs[i])
        loss = criterion(outputs, labels[i])
        loss.backward()
        optimizer.step()
        if print_f:
            print('loss:', loss.item())
        return loss.item()

def default_training(net, trainloader, num_epochs):
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            #inputs, labels = data.to(device)
            inputs, labels = [d.to(device) for d in data]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        return loss.item()

def test_acc(valloader, classes, num_classes):
    correct = 0
    total = 0
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    with torch.no_grad():
        # dataiter = iter(valloader)
        #for data in valloader:
        for i, data in enumerate(valloader, 0):
            images, labels = [d.to(device) for d in data] # data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            c = (predicted == labels).squeeze()

            for i in range(len(labels)):
                if len(labels) == 1:
                    label = labels[i]
                    class_correct[label] += c.item()
                else:
                    label = labels[i]
                    class_correct[label] += c[i].item()
                class_total[label] += 1

    accuracy = 100 * correct / total
    print('total images looked at:', total)
    print('Accuracy of the network on the %d test images: %d %%' % (len(valloader.dataset),
        100 * correct / total))

    class_percentage = []
    for i in range(num_classes):
        if class_total[i] == 0:
            print('Accuracy of %5s : 0 Examples' % (classes[i]))
            class_percentage.append(-1)
        else:
            print('Accuracy of %5s : %4d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
            class_percentage.append(100 * class_correct[i] / class_total[i])

    class_percentage = [class_correct[i] / class_total[i] for i in range(num_classes)]

    return accuracy, class_correct, class_total, class_percentage

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        return output

# ==================================================================================================================
print('Main Function running...')

trainSet, trainloader, valloader, testloader, classes = setup_database()

# Network Setup ----------------------------------------------------------------------------------------------------
net = Net()
print(net)

torch.cuda.empty_cache() # for emptying out CUDA cache
print('Cuda device:', device)
net.to(device)
criterion = nn.NLLLoss() #nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
# ------------------------------------------------------------------------------------------------------------------
sampling = 'largest_margin' # 'random', 'largest_margin', 'smallest_margin'
print('Sampling being used for active learning:', sampling)
loss = 1
training_epochs = 350
num_epoch = 0
num_batches = 25 # 3200 training examples from like 64 batch size
losses = []
accuracies = []
saveTime = time.time()

# let's train our model --------------------------------------------------------------------------------------------
while num_epoch < training_epochs:
    print_f = False
    if num_epoch % 20 == 0:
        print('Num_epoch: ', num_epoch)
        print_f = True
        print('loss:', loss)

        # Test on validation set
        accuracy, class_correct, class_total, class_percentage = test_acc(valloader, classes, num_classes)
        accuracies.append(accuracy)
        print('Class Totals:', class_total)

        class_percentage = np.asarray(class_percentage)
        class_percentageTitle  = 'class_percentage_' + str(num_epoch) + '_' + sampling + '_Train_' + str(saveTime)
        np.save(class_percentageTitle, class_percentage)

    data_act, labels_act = get_active_batches(trainloader, net, sampling, num_batches, print_f)
    loss = train(net, data_act, labels_act, print_f)
    #loss = default_training(net, trainloader, 1) # for just running on everything

    losses.append(loss)
    num_epoch += 1
print('Finished Training')
print('Epochs:', num_epoch)
# ------------------------------------------------------------------------------------------------------------------

# Save our Data ----------------------------------------------------------------------------------------------------
# save our neural net
PATH = './cub_birds_net_random' + str(saveTime) + '.pth'
torch.save(net.state_dict(), PATH)

# save our losses and accuracies -----------------------------------------------------------------------------------
losses = np.asarray(losses)
lossesTitle = 'losses_' + str(num_epoch) + '_' + sampling + '_Train_' + str(saveTime)
np.save(lossesTitle, losses)

accuracies = np.asarray(accuracies)
accuraciesTitle = 'accuracies_' + str(num_epoch) + '_' + sampling +'_Train_' + str(saveTime)
np.save(accuraciesTitle, accuracies)
# ------------------------------------------------------------------------------------------------------------------

# Get our test accuracy --------------------------------------------------------------------------------------------
print('Final Test on Test Set ==================================================')
accuracy, class_correct, class_total, class_percentage = test_acc(testloader, classes, num_classes)

class_percentage = np.asarray(class_percentage)
class_percentageTitle  = 'class_percentage_' + str(num_epoch) + '_' + sampling +'_Test_' + str(saveTime)
np.save(class_percentageTitle, class_percentage)
# ------------------------------------------------------------------------------------------------------------------
