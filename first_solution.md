# Student Name/ ID:

Aman Bhatta
Net_ID : 902186627


# Network Architecture and Summary

a) Layers: I have implemented ResNet 50/101/152 from scratch for the gender classification model. 

b) There could be several loss function available but the loss of preference for two class classification in general is binary cross-entropy loss. However, I have used a cross entropy loss function in the network and mainly because, I want to implement CAM to this network.

c) Optimizations: I have used standard Adam Optimizer, with initial learning rate of 0.1 with step reduction by factor of 0.1 every 15 epochs. 

d) Accuracy: To set the pipeline, I have used a Gender-Classification dataset that I got from Kaggle. The link for the dataset is: https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset. It has ~23,000 images for training and ~5,500 images for validation for each class. 

~~~bash
Training Accuracy at the end of Epoch 50 = 96.69425201416016%
Accuracy for the validation set = 95.9738998413086%
~~~
Log Files for the Training/Validation Accuracy can be found here: [log file](https://github.com/Czajka-Teaching/semester-project-abhatta1234/blob/main/ResNet/gender_classification_resnet.o337624)
# Future Task and Accuracy Improvements

a) I am planning to implement Squeeze and Excitations as mentioned in this paper: https://arxiv.org/pdf/1709.01507.pdf to boost up the accuracy.

b) ResNet model has Global Average Pooling layer after final convolution layer, so this model is by default compatible to generate CAM as in paper: http://cnnlocalization.csail.mit.edu/Zhou_Learning_Deep_Features_CVPR_2016_paper.pdf. So, I will modify ResNet model to generate CAM.

c) Train this complete network on MS1MV_2 dataset, which has ~3.8M images for males and ~1.8M images for females. I am planning to train the network on equal number of male/female images i.e 1.8M images each class.

d) Finally, test this image on MORPH. Also, generate CAM for MORPH dataset. Images for MORPH as aligned, so stacking heatmap for all males/females will most likely indicated what were the most salient region for gender classification. 


# Check the prediction for one validation image

To check the validation model, pass the num_layers and path to image to get results.

Note: Currently, only the pretrained version is available for ResNet-50.

~~~bash
python3 ResNet/python3 check_val.py -n num_layers -img path_to_img
~~~


# Train the network from Scratch

To run the model, two things are required:

1) Change the path to training data in main.py
2) Change the path to validation data in main.py


~~~bash
python3 ResNet/main.py -n num_layers
~~~

Notes:
1) ResNet model accepts [50,101,152] as the acceptable number of layers
2) For future update, the path to train and validation data will be available to be passed from the arguments
