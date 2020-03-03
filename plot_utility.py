import numpy as np
import matplotlib.pyplot as plt
import os

uncertainties = ['random', 'largest_margin', 'smallest_margin', 'least_confident', 'entropy']

data_dir = '/home/memo/Documents/senior/Winter/CNS_186/vision_project/CNS-186-Project/MNIST_trials'


def avg_plots(uncertainty):
    acc_data = []
    loss_data = []
    folder_dir = data_dir + '/' + uncertainty + '_MNIST_'

    for i in range(3):
        fold = folder_dir + str(i)
        files = os.listdir(fold)
        accuracy_file = ''
        loss_file = ''

        for file in files:
            if 'accuracies_350' in file and 'npy' in file:
                accuracy_file = file
            elif 'losses_350' in file and 'npy' in file:
                loss_file = file

        accuracies = np.load(fold + '/' + accuracy_file)
        losses = np.load(fold + '/' + loss_file)

        acc_data.append(accuracies)
        loss_data.append(losses)

    avg_acc = np.zeros(acc_data[0].shape)
    avg_loss = np.zeros(loss_data[0].shape)

    for i in range(len(acc_data)):
        avg_acc += acc_data[i]
        avg_loss += loss_data[i]

    avg_acc = avg_acc / len(acc_data)
    avg_loss = avg_loss / len(avg_loss)

    return [avg_acc, avg_loss]


dict_plots = {}
x_epochs = [i for i in range(0, 360, 20)]

for i in range(len(uncertainties)):
    avg_acc, avg_loss = avg_plots(uncertainties[i])
    dict_plots[uncertainties[i]] = [avg_acc, avg_loss]

    plt.plot(x_epochs, avg_acc, label=uncertainties[i])

plt.title('Compare Each Uncertainty Measure with Accuracy, Averaged Over 3 Runs')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy on Test Set')
plt.legend()
plt.show()
