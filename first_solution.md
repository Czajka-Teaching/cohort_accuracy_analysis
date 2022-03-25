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
