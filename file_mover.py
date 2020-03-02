import os 
import shutil 
from shutil import copyfile 

data_dir = "/home/memo/Documents/senior/Winter/CNS_186/vision_project/CUB_200_2011/CUB_200_2011"
train_dir = data_dir + "/images"

input_dir = train_dir
output_dir = data_dir + "/images_aug"

classes = []
f = open(data_dir + "/classes.txt", "r")
classes = []
counter = 1
for x in f:
  classes.append(x[(len(str(counter)) +1):-1])
  counter += 1
f.close()

for i in range(len(classes)):
    entries = os.listdir(train_dir + '/' + classes[i])
    entries_jpg = []
    for entry in entries:
        if '.jpg' in entry:
            entries_jpg.append(entry)

    for j in range(len(entries_jpg)):
        copyfile(input_dir + '/' + classes[i] + '/' + entries_jpg[j], output_dir + '/' + classes[i] + '/' + entries_jpg[j])
    print('Copied all of ', classes[i])
