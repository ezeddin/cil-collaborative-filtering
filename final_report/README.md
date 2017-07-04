How to run the code:
--------------------
To generate the best possible submission, use the command:

python train.py --model=SGDnn

which will first run the SGD+ model and then apply the neural postprocessor.
Because it is difficult to make keras produce reproducible results, we cannot guarantee that the submission file will be exactly the same as the one uploaded on kaggle. The score however should be similar, and the SGD+ model will always generate the same factorizations.


How to generate the plots:
--------------------------
Figure 1 : These plots were generated using the attached excel file (TODO join excel file). The data in the excel file was extracted from the cross-validation output

Figure 2 : This plot was generaded using the following routine:

  1. Firstly, the grid search has to be run using the `multi_job_submitter.py` script on Euler. This generates multiple jobs each running the `batch_train.py` script which itself runs multiple parallel processes that call `train.py` with a certain range of hyper-parameters. The `train.py` script stores its result as a numpy array in a pickle file called `data/scores_<timestamp>_<K_values>_<L_value>_<L2_value>.pkl`. However, the `batch_train.py` script also stores the results of the different runs of `train.py` as a numpy array in a pickle file called `data/grid_search_<timestamp>`. So in the end, one file for every job is created.
  2. The results of these files then have to be concatenated using the `batch_train_postconcatenate.py` script. This script generated the file `data/grid.mat` which contains all results in different rows with the columns being _K_, _L_, _L2_ and the score.
  3. This file can then be loaded with the `grid_plot.m` MATLAB script which finally displays the plot.

Figure 3 : This plot was produced by running the MATLAB script `dropout_interpolation.m` which contains the scores from the kaggle website.
