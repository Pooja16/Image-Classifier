# Developing an Image Classifier with Deep Learning

## Introduction:

The goal was to train an image classifier to recognize different species of flowers. Used Udacity’s workspace enabled with GPU for working on this project.

##  Table of Contents:

This project contains the following files:

- `Image Classifier Project.ipynb` : This python file has all the code that was used to put through the entire analysis. 
- `train.py` : This includes the python code used to train the deep learning classifier. 
- `predict.py` : This has the code use to predict the images. 
- `program.py` : This contatins all the functions that are used by train.py and predict.py
- `output_log.txt`: This contains the output log.

In the Terminal or Command Prompt, navigate to the folder containing the project files, and then use the command `jupyter notebook Image Classifier Project.ipynb ` to open up a browser window or tab to work with your notebook. Alternatively, you can use the command `jupyter notebook` or `ipython notebook` and navigate to the notebook file in the browser window that opens.

##  Summary of Analysis:

The process is broken into multiple steps

- Load and preprocess the image dataset
Applied transformations such as random scaling, cropping and flipping. Have split dataset into training, validation and test data set.

- Building and training the classifier
Loaded a pre-trained network i.e. VGG16 network. Defined a new untrained feed forward network as a classifier, using ReLu activations and dropout. Trained the classifier using the pre-trained network to get the features. Tracked the loss and accuracy on the validation set to determine the best hyper parameters.

- Save and load the checkpoint
As the network has been trained, the model was saved so that it can be loaded later for making predictions.

- Class Prediction
Used the trained network for interpreting the inference. Pass an image into the network and predict the class of the image.


##  Software:

This project uses the following software and Python libraries:
-  [Python 2.7] (https://www.python.org/download/releases/2.7/)
-  [NumPy] (http://www.numpy.org/)
-  [Pandas] (http://pandas.pydata.org/)
-  [scikit-learn] (http://scikit-learn.org/stable/)
-  [matplotlib] (http://matplotlib.org/)

You will also need to have software installed to run and execute a [Jupyter Notebook] (http://ipython.org/notebook.html)
If you do not have Python installed yet, it is highly recommended that you install the [Anaconda] (http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. Make sure that you select the Python 2.7 installer and not the Python 3.x installer.
