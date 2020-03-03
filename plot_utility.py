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

def class_performance_data(uncertainty):
    folder_dir = data_dir + '/' + uncertainty + '_MNIST_'

    # since it may be arbitrary which class performance got boosted, we're not
    # going to average these
    acc_data_runs = []
    for i in range(3):
        fold = folder_dir + str(i)
        files = os.listdir(fold)

        acc_data = None

        acc_found_flag = False
        accuracy_file = 'class_percentage_0_'
        for file in files:
            if accuracy_file in file:
                print('accuracy file found')
                acc_data = np.load(fold + '/' + file)
                acc_found_flag = True

        if not acc_found_flag:
            import pdb; pdb.set_trace()

        for epoch in range(20,360,20):
            fileStr = 'class_percentage_' + str(epoch)
            for file in files:
                if fileStr in file:
                    new_col = np.load(fold + '/' + file)
                    acc_data = np.column_stack((acc_data, new_col))

        acc_data_runs.append(acc_data)

    return acc_data_runs


def generate_acc_class_plots():
    acc_data_uncertain = {}
    epochs_x = [i for i in range(0, 360, 20)]
    epochs_x.append(350)

    for i in range(len(uncertainties)):
        acc_data_runs = class_performance_data(uncertainties[i])
        acc_data_uncertain[uncertainties[i]] = acc_data_runs

        for j in range(len(acc_data_runs)):
            print(len(acc_data_runs))
            for k in range(acc_data_runs[i].shape[0]):
                print(acc_data_runs[i].shape)
                plt.plot(epochs_x, acc_data_runs[i][k,:], label='Class ' + str(k))

            title = 'Class Accuracy Over Epochs for \n Uncertainty Measure: ' + uncertainties[i]
            titleFile = 'Class_Acc_' + uncertainties[i] + '_run_' + str(j) + '.png'
            plt.title(title)
            plt.xlabel('Number of Epochs')
            plt.ylabel('Accuracy on Validation Set')
            plt.legend()
            plt.savefig('./plots/' + titleFile)
            plt.show()

    return acc_data_uncertain

def smooth_plot(data, num_avg=5):
    result_data = []
    for i in range(0, len(data) - num_avg, num_avg):
        result_data.append(sum(data[i:i+num_avg]) / num_avg)

    return result_data

def generate_acc_loss_plots(acc = True):
    colors = ['black', 'blue', 'red', 'orange', 'magenta']
    dict_plots = {}
    x_epochs = [i for i in range(0, 360, 20)]

    title = 'MNIST_Loss_3_Runs_Avg_Smooth.png'

    if acc:
        title = 'MNIST_Acc_3_Runs_Avg.png'

    for i in range(len(uncertainties)):
        avg_acc, avg_loss = avg_plots(uncertainties[i])
        dict_plots[uncertainties[i]] = [avg_acc, avg_loss]

        if acc:
            plt.plot(x_epochs, avg_acc, label=uncertainties[i], c=colors[i], linewidth=2)
        else:
            avg_loss_smooth = smooth_plot(avg_loss, 5)
            x_epochs = [i for i in range(0, 340 + 5, 5)]

            #plt.plot(x_epochs, avg_loss, label=uncertainties[i])
            plt.plot(x_epochs, avg_loss_smooth, label=uncertainties[i], c=colors[i], linewidth=2)


    plt.title('Compare Each Uncertainty Measure with Accuracy \n Averaged Over 3 Runs')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss on Training Set')
    plt.legend()
    plt.savefig('./plots/' + title)
    plt.show()


generate_acc_loss_plots()
#generate_acc_class_plots()
