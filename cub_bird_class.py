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
    transform = transforms.Compose(
        [transforms.Resize((400,400), interpolation=Image.NEAREST),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data_dir = '../cub_birds_data/'
    dataSet = torchvision.datasets.ImageFolder(root=data_dir+'images', transform=transform)

    num_train = int(len(dataSet) * .7)
    num_val = int((len(dataSet) - num_train) / 2)
    num_test = len(dataSet) - num_train - num_val

    trainSet, val_testSetTmp = torch.utils.data.random_split(dataSet, [int(num_train), int(num_val + num_test)])
    valSet, testSet = torch.utils.data.random_split(val_testSetTmp, [int(num_val), int(num_test)])

    trainloader = torch.utils.data.DataLoader(dataSet, batch_size=8,
                                              shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valSet, batch_size=8,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testSet, batch_size=8,
                                             shuffle=True, num_workers=2)

    f = open(data_dir + "lists/classes.txt", "r")
    classes = []
    for x in f:
      classes.append(x[4:-1])

    print(type(dataSet))
    print(type(trainSet))
    print(type(trainloader))

    return (trainSet, trainloader, valloader, testloader, classes)

# same uncertainty measure as seen in that medium post
# uncertainty = most confident - least confident
# we want to put the smallest uncertainties in our training
# because that means that that sample is not well defined

def get_active_batches(trainloader, net, num_batches=1, print_f=False):
    flag = True
    #data_active = None
    #labels_active = None
    #uncertain_ratio = None
    act_dict = {}
    count_key = 0

    for i, data in enumerate(trainloader, 0):
        inputs, labels = [d.to(device) for d in data]

        outputs = net(inputs)

        outputs_cpu = outputs.cpu()
        outputs_np = outputs_cpu.detach().numpy()

        max_batch = 0
        min_batch = 0
        for out in outputs_np:
            max_batch += max(out)
            min_batch += min(out)

        if len(act_dict.keys()) < num_batches:
            act_dict[count_key] = [max_batch - min_batch, inputs, labels]
            count_key += 1
        else:
            for key in act_dict:
                if act_dict[key][0] > max_batch - min_batch:
                    act_dict[key] = [max_batch - min_batch, inputs, labels]

    if print_f:
        print('largest uncertainties', [act_dict[key][0] for key in act_dict])
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

def test_acc(valloader, classes, num_classes):
    correct = 0
    total = 0
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    with torch.no_grad():
        # dataiter = iter(valloader)
        for data in valloader:
            images, labels = [d.to(device) for d in data] # data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            c = (predicted == labels).squeeze()

            for i in range(len(labels)):
                if len(labels) == 1:
                    #import pdb; pdb.set_trace()
                    label = labels[i]
                    class_correct[label] += c.item()
                else:
                    label = labels[i]
                    class_correct[label] += c[i].item()
                class_total[label] += 1

    accuracy = 100 * correct / total
    print('Accuracy of the network on the %d test images: %d %%' % (len(valloader),
        100 * correct / total))

    for i in range(num_classes):
        print('Accuracy of %5s : %4d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))

    class_percentage = [class_correct[i] / class_total[i] for i in range(num_classes)]

    return accuracy, class_correct, class_total, class_percentage


# ==================================================================================================================
print('Main Function running...')
trainSet, trainloader, valloader, testloader, classes = setup_database()

# Network Setup ----------------------------------------------------------------------------------------------------
net = models.resnet18()
#net.fc = torch.nn.Linear(512, 200)
#net.fc = torch.nn.Linear(512, 10) # trying smaller dataset
net.fc = torch.nn.Linear(512, num_classes)
net = nn.Sequential(
    net,
    nn.Softmax(1)
)

torch.cuda.empty_cache() # for emptying out CUDA cache
print('Cuda device:', device)
net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# ------------------------------------------------------------------------------------------------------------------

loss = 1
training_epochs = 350
num_epoch = 0
losses = []
accuracies = []
saveTime = time.time()

# let's train our model --------------------------------------------------------------------------------------------
while loss > 0.2 and num_epoch < training_epochs:
    print_f = False
    if num_epoch % 20 == 0:
        print('Num_epoch: ', num_epoch)
        print_f = True

        # Test on validation set
        accuracy, class_correct, class_total, class_percentage = test_acc(valloader, classes, num_classes)
        accuracies.append(accuracy)
        class_correct = np.asarray(class_correct)
        class_total = np.asarray(class_total)
        class_percentage = np.asarray(class_percentage)

        class_correctTitle = 'class_correct_' + str(num_epoch) + '_activeLearnVal_' + str(saveTime)
        class_totalTitle  = 'class_total_' + str(num_epoch) + '_activeLearnVal_' + str(saveTime)
        class_percentageTitle  = 'class_percentage_' + str(num_epoch) + '_activeLearnVal_' + str(saveTime)

        np.save(class_correctTitle, class_correct)
        np.save(class_totalTitle, class_total)
        np.save(class_percentageTitle, class_percentage)

    data_act, labels_act = get_active_batches(trainloader, net, 5, print_f)
    loss = train(net, data_act, labels_act, print_f)
    losses.append(loss)
    num_epoch += 1
print('Finished Training')
print('Epochs:', num_epoch)
# ------------------------------------------------------------------------------------------------------------------

# Save our Data ----------------------------------------------------------------------------------------------------
# save our neural net
PATH = './cub_birds_net_' + str(saveTime) + '.pth'
torch.save(net.state_dict(), PATH)

# save our losses and accuracies
losses = np.asarray(losses)
lossesTitle = 'losses_' + str(num_epoch) + '_activeLearn_' + str(saveTime)
np.save(lossesTitle, losses)

accuracies = np.asarray(accuracies)
accuraciesTitle = 'accurcies_' + str(num_epoch) + '_activeLearn_' + str(saveTime)
np.save(accuraciesTitle, accuracies)
# ------------------------------------------------------------------------------------------------------------------

# Get our test accuracy --------------------------------------------------------------------------------------------
accuracy, class_correct, class_total, class_percentage = test_acc(testloader, classes, num_classes)
class_correct = np.asarray(class_correct)
class_total = np.asarray(class_total)
class_percentage = np.asarray(class_percentage)

class_correctTitle = 'class_correct_' + str(num_epoch) + '_activeLearnTest_' + str(saveTime)
class_totalTitle  = 'class_total_' + str(num_epoch) + '_activeLearnTest_' + str(saveTime)
class_percentageTitle  = 'class_percentage_' + str(num_epoch) + '_activeLearnTest_' + str(saveTime)

np.save(class_correctTitle, class_correct)
np.save(class_totalTitle, class_total)
np.save(class_percentageTitle, class_percentage)
# ------------------------------------------------------------------------------------------------------------------

import pdb; pdb.set_trace()
