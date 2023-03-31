<h1>Deep Learning - Ex.1</h1>
The code is written using pytorch and is divided to 3 files:

- main.py
- models.py
- trainer.py

<b>main.py </b> - loads the data and call training and testing for all models. Usage:
- Train a model by setting 'use_saved_weights=False' (with plot) or delete the model's pkl file.
- Check a loaded model accuracy by setting 'use_saved_weights=True' (accuracy print only, without plot), if the pkl file is
missing the program will train the model first.

<b>models.py </b>- classes of different models (original net, net with dropout, net with BN)

<b>trainer.py </b>- trains & tests a model