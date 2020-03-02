import os
import sys
import numpy as np

# split data into train, val, and test

data_dir = "/home/memo/Documents/senior/Winter/CNS_186/vision_project/CUB_200_2011/CUB_200_2011"
train_dir = data_dir + "/images"
val_dir = data_dir + "/images_val"
test_dir = data_dir + "/images_test"

classes = []
f = open(data_dir + "/classes.txt", "r")
classes = []
counter = 1
for x in f:
  classes.append(x[(len(str(counter)) +1):-1])
  counter += 1
f.close()

print("classes: \n", classes)

for i in range(len(classes)):
    entries = os.listdir(train_dir + '/' + classes[i])
    entries_jpg = []
    for entry in entries:
        if '.jpg' in entry:
            entries_jpg.append(entry)

    print('number entries: %d number entries jpg %d' %(len(entries), len(entries_jpg)))

    num_train = int(len(entries_jpg) * .7)
    num_val =  int((len(entries_jpg) - num_train) / 2)
    num_test = len(entries_jpg) - num_train - num_val

    print('num_train: %d, num_val %d, num_test %d' % (num_train, num_val, num_test))

    val_test_entries = np.random.choice(entries_jpg, (num_val + num_test), replace=False)
    val_entries = val_test_entries[:num_val]
    test_entries = val_test_entries[num_val:]

    print('val_entries: \n', val_entries)
    print('test_entries: \n', test_entries)

    if not os.path.exists(val_dir+'/'+classes[i]):
        os.mkdir(val_dir+'/'+classes[i])
    if not os.path.exists(test_dir+'/'+classes[i]):
        os.mkdir(test_dir+'/'+classes[i])

    for j in range(len(val_entries)):
        os.replace(train_dir + '/' + classes[i] + '/' + val_entries[j], val_dir + '/' + classes[i] + '/' + val_entries[j])

    for j in range(len(test_entries)):
        os.replace(train_dir + '/' + classes[i] + '/' + test_entries[j], test_dir + '/' + classes[i] + '/' + test_entries[j])

    print('Data Split for: ', classes[i])
