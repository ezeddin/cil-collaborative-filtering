collaborative filtering
=======================

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



Euler Cheat Sheet
=================

This short description is a summary of the [wiki](https://scicomp.ethz.ch/wiki/Getting_started_with_clusters).


Connection
----------

Login with `ssh <username>@euler.ethz.ch`.
To do file operations and also be able to edit file contents with a GUI, I recommend also connecting with WinSCP.
This way, source files or scripts can be edited by just double clicking on them in the WinSCP explorer window.

Note that outside the ETH, you have to be connected through VPN to get access to Euler.


File Organization
-----------------

On euler you have a home directory at `/cluster/home/<username>`. Set the file up as follows:

1. clone the git repository to a subfolder (the exact remote url string is important to be able to sign there in as collaborator):
   `git clone https://<YOUR_github_username>@github.com/<path_to_repository>.git`
2. add data files not tracked by git with WinSCP (drag and drop).


Python Usage
------------

To make python available as a module, call `module load python/3.3.3`.
Now you can test your environment by trying to run a python file.


Job Submission
--------------

To submit a job to the queue system, call `bsub <command> [<arguments>]`. If your job exceed the default resource allocation (in terms of memory, computation time, hard drive space etc), you have to pass your resource requirements:

Most important, if your job will take more than 4 hours to complete, submit it by passing it the expected duration (and some margin...) `bsub -W <hours>:<minutes>`. Other resource requirement parameters like number of CPU cores as well as switches for email notification etc. are available in [submission command configurator](https://scicomp.ethz.ch/lsf_submission_line_advisor/)


Monitoring and Evaluation
-------------------------

Once submitted, every job gets an ID assigned and awaits execution in a "pending" queue.
You can watch the state and resources of your current jobs (still pending for execution or already running) with `bjobs` and `bbjobs`.
To see the `STDOUT` stream of a running job, you can type `bpeek <JOB_ID>`.

You can even connect to the machine on which the job is running by `bjob_connect <JOB_ID>`.
Running jobs as well as pending jobs can be killed with `bkill <JOB_ID>`.

If the job finished, a log file containing the complete `STDOUT` stream is written called `lsf.o<JOB_ID>`.


Example
-------

I run my python script using
`bsub -n 1 -W 00:30 -N -B -R "rusage[mem=512]" "python3 train.py --model=SGD --cv_splits=8 --param=list(range(2,20)) --n_iter=20000000"`
