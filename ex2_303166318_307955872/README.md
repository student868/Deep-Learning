<h1>Deep Learning - Ex.2</h1>

The code is written using pytorch and is divided to several files:

<b>notebook.ipynb </b> - Jupyter notebook for calling functions from main.py. Usage:

1. Load the data
- <b>Notice:</b> scipts, models and data are downloaded from our Github, because mounting from google drive on a shared folder made us problems, nevertheless, all the files are identical to those in the drive if you want to copy them and mount (you can verify that the last commit of the folder was made before the submission date).
2. Train / evaluate saved model by running one of the following functions as instructed:

- LSTM
- LSTM with dropout
- GRU
- GRU with dropout

Tips:
- Training a model prints the training procedure and plot the graph. <br />
- Evaluating the saved model prints only the accuracy of the saved model on the data (if the pkl file is missing the program will train the model first).
- If you want to overwrite the saved model uncomment "torch.save" call in "evaluate_model" function in "main.py" and train the model (<b>Warning:</b> the saved model will be deleted).

<b>main.py </b> - load the data and call training and testing functions.

<b>trainer.py </b>- model & train & test implementation.

<b>data </b> (folder) - contains the PTB data (downloaded from moodle).

<b>models </b> (folder) - contains the saved models.

<b>plots </b> (folder) - contains the plots.

<br /><br />
Sources:
- [Recurrent Neural Network Regularization](https://arxiv.org/abs/1409.2329)
- [Paper github repo](https://github.com/wojzaremba/lstm)
- [Another Github repo](https://github.com/ahmetumutdurmus/zaremba/blob/ac4127dce7f955bf291e430ea7689c0db027ae69/main.py)
- [Kaggle Tutorial](https://www.kaggle.com/code/beastlyprime/pytorch-beginner-language-model/notebook)
