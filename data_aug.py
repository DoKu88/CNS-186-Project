import numpy as np
import Augmentor

input_dir = '/home/memo/Documents/senior/Winter/CNS_186/vision_project/cub_birds_data/images_aug'
output_dir = '/home/memo/Documents/senior/Winter/CNS_186/vision_project/cub_birds_data/images_aug1'
data_dir = '/home/memo/Documents/senior/Winter/CNS_186/vision_project/cub_birds_data'

classes = []
f = open("/home/memo/Documents/senior/Winter/CNS_186/vision_project/cub_birds_data/lists/classes.txt", "r")
classes = []
for x in f:
  classes.append(x[:-1])
f.close()

print(classes)

for i in range(10):

    p = Augmentor.Pipeline(source_directory= input_dir + '/' + classes[i],
                            output_directory = output_dir + '/' + classes[i])

    #p.flip_left_right(probability=1.0)
    p.rotate(probability=0.9, max_left_rotation=15, max_right_rotation=15)
    p.shear(probability=0.9, max_shear_left=20, max_shear_right=20)
    #p.zoom(probability = 0.9, min_factor = 1.1, max_factor = 1.6)
    p.process()
