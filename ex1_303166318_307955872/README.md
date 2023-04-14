<h1>Deep Learning - Ex.1</h1>

We used the pytorch Quickstart guide: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

The code is written using pytorch and is divided to 3 files:

- main.py
- models.py
- trainer.py

<b>notebook.ipynb </b> - Jupyter notebook for calling functions from main.py. Usage:

1. Load the data.
2. Train / evaluate saved model by running one of the following functions as instructed:

- train_original
- train_dropout
- train_weight_decay
- train_batch_normalization

Tips:
- Training a model prints the training procedure and plot the graph. <br />
- Evaluating the saved model prints only the accuracy of the saved model on the data (if the pkl file is missing the program will train the model first).
- If you want to overwrite the saved model uncomment "torch.save" call in "evaluate_model" function in "main.py" and train the model (<b>Warning:</b> the saved model will be deleted).

<b>main.py </b> - load the data and call training and testing functions.

<b>models.py </b>- classes of different models (original net, net with dropout, net with BN).

<b>trainer.py </b>- train & test a model (pytorch code).

<b>data </b> (folder) - contains the Fashion MNIST data.

<b>models </b> (folder) - contains the saved models.

<b>plots </b> (folder) - contains the plots.