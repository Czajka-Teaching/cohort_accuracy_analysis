# Network Architecture and Summary

a) Layers: I have implemented ResNet 50/101/152 from scratch for the gender classification model. 

b) There could be several loss function available but the loss of preference for two class classification in general is binary cross-entropy loss. However, I have used a cross entropy loss function in the network and mainly because, I want to implement CAM to this network.

c) Optimizations: I have used standard Adam Optimizer, with initial learning rate of 0.1 with step reduction by factor of 0.1 every 15 epochs. 

d) Accuracy: 










# Run the network

To run the model, two things are required:

1) Change the path to training data in main.py
2) Change the path to validation data in main.py


~~~bash
python3 ResNet/main.py -n num_layers
~~~

Notes:
1) ResNet model accepts [50,101,152] as the acceptable number of layers
2) For future update, the path to train and validation data will be available to be passed from the arguments
