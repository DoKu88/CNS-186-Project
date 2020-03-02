import numpy as np
import Augmentor
import os

data_dir = "/home/memo/Documents/senior/Winter/CNS_186/vision_project/CUB_200_2011/CUB_200_2011"
train_dir = data_dir + "/images"
val_dir = data_dir + "/images_val"
test_dir = data_dir + "/images_test"

input_dir = data_dir + "/images_aug"
output_dir = data_dir + "/images_aug1"

classes = []
f = open(data_dir + "/classes.txt", "r")
classes = []
counter = 1
for x in f:
  classes.append(x[(len(str(counter)) +1):-1])
  counter += 1
f.close()

print(classes)

for i in range(len(classes)):

    if not os.path.exists(output_dir + '/' + classes[i]):
        os.mkdir(output_dir + '/' + classes[i])

    print('output_directory', output_dir + '/' + classes[i])

    p = Augmentor.Pipeline(source_directory= input_dir + '/' + classes[i],
                            output_directory = output_dir + '/' + classes[i])

    #p.flip_left_right(probability=1.0)
    p.rotate(probability=.95, max_left_rotation=15, max_right_rotation=15)
    p.shear(probability=.95, max_shear_left=20, max_shear_right=20)
    #p.zoom(probability = 0.9, min_factor = 1.1, max_factor = 1.5)
    p.process()
