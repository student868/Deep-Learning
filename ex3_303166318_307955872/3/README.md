<h1>Deep Learning - Ex.3 Question 3</h1>

The code is written using pytorch and is divided to several files:

<b>notebook.ipynb </b> - Jupyter notebook for calling functions from main.py. Usage:

1. Load imports
- <b>Notice:</b> scipts, models and data are downloaded from our Github, because mounting from google drive on a shared folder made us problems, nevertheless, all the files are identical to those in the drive if you want to copy them and mount (you can verify that the last commit of the folder was made before the submission date).
2. Train / evaluate saved model by running one of the following functions as instructed:

- MNIST
- Fashion MNIST

Tips:
- For each dataset you can choose between training and evaluating a saved model. 
- If you want to overwrite the saved model uncomment "torch.save" call in "train" function in "main.py" and train the model (<b>Warning:</b> the saved model will be deleted).

<b>main.py </b> - load the data and call training and testing functions.

<b>models.py </b>- class of the VAE.

<b>trainer.py </b>- model & train & test implementation.

<b>data </b> (folder) - contains the data.

<b>models </b> (folder) - contains the saved models.


Sources:
- [Semi-supervised Learning with Deep Generative Models](https://arxiv.org/abs/1406.5298)
- [Official implementation](https://github.com/dpkingma/nips14-ssl/blob/master/gpulearn_z_x.py)
- [VAE Tutorial](https://sannaperzon.medium.com/paper-summary-variational-autoencoders-with-pytorch-implementation-1b4b23b1763a)
