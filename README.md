# CNS-186-Project
## Active Learning for Image Classification
For this project, we are going to train a Resnet-18 Architecture to classify
images from the Caltech-UCSD Birds 200 2011 (CUB-200) dataset. 

With this, we are going to train using both active and standard supervised 
learning techniques to compare how these differ for image classification. 
Additionally, we are going to see if differing how we choose which data
to train on for our active learning model affects our model and if we 
can incorporate hierarchical data about the image classes (bird species) to
improve our classification. 

Note: Didn't put Torch version to use in Requirements.txt, since you already
may have something set up. Please check out the following detailed links about
what version you should use of torch with your cuda version:
https://pytorch.org/get-started/previous-versions/
https://pytorch.org/
