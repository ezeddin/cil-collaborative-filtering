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