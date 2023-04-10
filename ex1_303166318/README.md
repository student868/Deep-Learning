<h1>Deep Learning - Ex.1</h1>

We used the pytorch Quickstart guide: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html

The code is written using pytorch and is divided to 3 files:

- main.py
- models.py
- trainer.py

<b>notebook.ipynb </b> - Jupyter notebook for calling functions from main.py. Usage:
1. Load the data.
2. Train / evaluate the model by calling one of the following functions:
  - train_original(train_dataloader, test_dataloader, use_saved_weights=True)
  - train_dropout(train_dataloader, test_dataloader, use_saved_weights=True)
  - train_weight_decay(train_dataloader, test_dataloader, use_saved_weights=True)
  - train_batch_normalization(train_dataloader, test_dataloader, use_saved_weights=True)


  - 'use_saved_weights=False' or delete the model's pkl file (accuracy print, with plot).
  - Check a loaded model accuracy by setting 'use_saved_weights=True' (accuracy print only, without plot), if the pkl file is
missing the program will train the model first.

<b>main.py </b> - load the data and call training and testing functions.

<b>models.py </b>- classes of different models (original net, net with dropout, net with BN).

<b>trainer.py </b>- train & test a model (pytorch code).