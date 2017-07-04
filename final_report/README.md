Task Description
================

A recommender system is concerned with presenting items (e.g. books on Amazon, movies at Movielens or music at lastFM) that are likely to interest the user. In collaborative filtering, we base our recommendations on the (known) preference of the user towards other items, and also take into account the preferences of other users.

Resources
---------
All the necessary resources (including training data) are available at https://inclass.kaggle.com/c/cil-collab-filtering-2017

Training Data
-------------
For this problem, we have acquired ratings of 10000 users for 1000 different items. All ratings are integer values between 1 and 5 stars.

Evaluation Metrics
------------------
Your collaborative filtering algorithm is evaluated according to the following weighted criteria:

- prediction error, measured by root-mean-squared error (RMSE)


How to run the code:
====================
To generate the best possible submission, use the command:

`python train.py --model=SGDnn`

which will first run the SGD+ model and then apply the neural postprocessor.
Making `Keras` produce reproducible results is known to be very difficult. Thus we cannot guarantee that the submission file will be exactly the same as the one uploaded on Kaggle. The score however should be similar.

To use the SGD+ model, use the command:

`python train.py`

This model will always generate the same factorizations.

The submissions files will be written to the submissions folder.


How to generate the plots:
==========================

Figure 1 : These plots were generated using the attached excel file (TODO join excel file). The data in the excel file was extracted from the cross-validation output

Figure 2 : This plot was generaded using the following routine:

  1. Firstly, the grid search has to be run using the `multi_job_submitter.py` script on Euler. This generates multiple jobs each running the `batch_train.py` script which itself runs multiple parallel processes that call `train.py` with a certain range of hyper-parameters. The `train.py` script stores its result as a numpy array in a pickle file called `data/scores_<timestamp>_<K_values>_<L_value>_<L2_value>.pkl`. However, the `batch_train.py` script also stores the results of the different runs of `train.py` as a numpy array in a pickle file called `data/grid_search_<timestamp>`. So in the end, one file for every job is created.
  2. The results of these files then have to be concatenated using the `batch_train_postconcatenate.py` script. This script generated the file `data/grid.mat` which contains all results in different rows with the columns being _K_, _L_, _L2_ and the score.
  3. This file can then be loaded with the `grid_plot.m` MATLAB script which finally displays the plot.

Figure 3 : This plot was produced by running the MATLAB script `dropout_interpolation.m` which contains the scores from the kaggle website.


Advanced Code Usage:
====================
``` 
usage: train.py [-h] [--submission SUBMISSION] [--model MODEL]
                [--cv_splits CV_SPLITS] [--score_averaging SCORE_AVERAGING]
                [--K K] [--L L] [--L2 L2] [--lr_factor LR_FACTOR] [--v V]
                [--n_messages N_MESSAGES] [--external_matrix EXTERNAL_MATRIX]
                [--model_path MODEL_PATH] [--save_model SAVE_MODEL]
                [--dropout DROPOUT] [--early_stopping EARLY_STOPPING]

optional arguments:
  -h, --help            show this help message and exit
  --submission SUBMISSION
                        Omit local cross-validation, just train the model and
                        export submission file. (default: True)
  --model MODEL         Prediction algorithm: average, SVD, SGD, SGD+, SGDnn
                        (default: SGD+)
  --cv_splits CV_SPLITS
                        Number of data splits for cross validation (default:
                        14)
  --score_averaging SCORE_AVERAGING
                        How many splits should be used as test data (default:
                        1)
  --K K                 First hyperparameter : Latent feature dimension size.
                        int or list (default: 12)
  --L L                 Second hyperparameter : Regularizer for SGD. (default:
                        0.083)
  --L2 L2               Third hyperparameter : Bias regularizer for SGD+
                        (default: 0.04)
  --lr_factor LR_FACTOR
                        Multiplier for the learning rate. (default: 3.0)
  --v V                 Verbosity of sgd algorithm: 0 for none, 1 for basic
                        messages, 2 for steps (default: 2)
  --n_messages N_MESSAGES
                        The number of messages to print for the sgd. Only
                        relevant when --v==2 (default: 20)
  --external_matrix EXTERNAL_MATRIX
                        In a multiprocessing environment: get matrices from
                        external arguments (default: False)
  --model_path MODEL_PATH
                        load matrices from external arguments (use to skip
                        SGD+ when using a neural network postprocessing)
                        (default: None)
  --save_model SAVE_MODEL
                        save all matrices (default: False)
  --dropout DROPOUT     Dropout value to use in neural network postprocessing.
                        (default: 0.6)
  --early_stopping EARLY_STOPPING
                        Observes the loss and stops early if it doesn't
                        decrease. (default: False)
```

Plot Script Usage
=================

The plotting of the grid searches worked as follows:

1. We first ran the grid search using the `multi_job_submitter.py` script on Euler. This generated multiple jobs running the `batch_train.py` script which itself ran multiple parallel processes that called `train.py` with the according hyper-parameter. The `batch_train.py` script stored the results of the different runs as a numpy array in a pickle file called `data/grid_search_<timestamp>`. So in the end, one file for every job was created.
2. The results of these files were then concatenated using the `batch_train_postconcatenate.py` script. This script generated the file `data/grid.mat` which contains all results in rows with the columns being _K_, _L_, _L2_ and the score.
3. This file could then be loaded with the `grid_plot.m` MATLAB script which finally displays the plot.

Authors
=======

Ezeddin Al Hakim, Cyril Stoller, Tobias Verhulst
