<h1>Deep Learning - Ex.3 Question 4</h1>

The code is written using pytorch and is divided to several files:

<b>notebook.ipynb </b> - Jupyter notebook for calling functions from main.py. Usage:

1. Load imports
<br /><b>Notice:</b> scripts, models and data are downloaded from our GitHub, because mounting from Google Drive on a shared folder made us problems, nevertheless, all the files are identical to those in the drive if you want to copy them and mount (you can verify that the last commit of the folder was made before the submission date).
2. Train / evaluate saved models by running the functions inside the notebook.

Tips:
- <b>It is recommended to use a GPU for training new models. </b>
- If you want to overwrite the saved model pass the parameter "save_trained_model=True" to "update_models" function call (<b>Warning:</b> the saved model will be deleted).

<b>main.py </b> - load the data and call training and testing functions.

<b>models.py </b>- classes of generator & discriminator.

<b>trainer.py </b>- model & train & test implementation.

<b>data </b> (folder) - contains the data.

<b>models </b> (folder) - contains the saved models.


Sources:
- [Improved Training of Wasserstein GANs (Paper)](https://arxiv.org/pdf/1704.00028.pdf)
- [Official implementation (TensorFlow)](https://github.com/igul222/improved_wgan_training/tree/master)
