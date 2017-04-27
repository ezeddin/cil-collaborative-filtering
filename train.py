# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse
import time
from local_vars import USERNAME
import pickle
import matplotlib.pyplot as plt
#%matplotlib inline

DATA_FILE = 'data/data_train.csv'
SUBMISSION_EXAMPLE = 'data/sampleSubmission.csv'
TARGET_FOLDER = 'submissions'
SUBMISSION_FORMAT='r{}_c{},{:.3f}\n'

NB_USERS = 10000
NB_ITEMS = 1000

DO_LOCAL_VALIDATION = True
MODEL = 'SVD'
INJECT_TEST_DATA = False
ROUND = 0

def load_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        nb_lines = len(lines) - 1
        
        rows = np.empty(nb_lines, dtype='int')
        cols = np.empty(nb_lines, dtype='int')
        values = np.empty(nb_lines, dtype='float')
        for i, l in enumerate(lines[1:]):
            s = l.split(',')
            values[i] = float(s[1].strip())
            c = s[0].split('_')
            rows[i] = int(c[1][1:]) - 1 # 0-based indexing
            cols[i] = int(c[0][1:]) - 1 # 0-based indexing
    data = scipy.sparse.csr_matrix((values, (rows, cols)), shape=(NB_ITEMS, NB_USERS), dtype='float')
    nb_ratings = (data!=0).sum()
    print('Dataset has {} non zero values'.format(nb_ratings))
    print('average rating : {}'.format( data.sum() / nb_ratings))
    return data.todense(), nb_ratings

def build_validation_set(raw_data):
    # randomly sample half of the non-zero values and define them as the new secret set
    non_zero_indices = list(zip(*np.nonzero(raw_data)))
    np.random.shuffle(non_zero_indices)
    secret_set_indices = non_zero_indices[:nb_ratings//2]
    known_set_indices = non_zero_indices[nb_ratings//2:]
    training_data = np.array(raw_data)
    training_data[list(zip(*secret_set_indices))] = 0
    secret_data = np.array(raw_data)
    secret_data[list(zip(*known_set_indices))] = 0
    return training_data, secret_data

def write_data(filename, matrix):
    with open(filename, 'w') as write_file:
        write_file.write('Id,Prediction\n')
        with open(SUBMISSION_EXAMPLE, 'r') as example_file:
            lines = example_file.readlines()
            
            for i, l in enumerate(lines[1:]):
                s = l.split(',')
                #values[i] = float(s[1].strip())
                c = s[0].split('_')
                user = int(c[0][1:]) - 1 # 0-based indexing
                item = int(c[1][1:]) - 1 # 0-based indexing
                
                rating = max(0, min(5, np.round(matrix[item,user], ROUND)))
                write_file.write(SUBMISSION_FORMAT.format(user+1, item+1, rating))

def average_prediction(matrix):
    average_for_item = np.true_divide(matrix.sum(1), (matrix!=0).sum(1))
    averaged = np.array(matrix, np.float64)
    for i in range(matrix.shape[0]):
        averaged[i,:] = np.where(averaged[i,:] == 0, average_for_item[i], averaged[i,:])
    return averaged

def svd_prediction(matrix, K=15):
    U, S, VT = np.linalg.svd(matrix, full_matrices=True)
    U2 = U[:,:K]
    S2 = S[:K]
    VT2 = VT[:K, :]
    return U2.dot(np.diag(S2)).dot(VT2)

def sgd_prediction(matrix):
    return np.array(matrix)

def validate(data, training_data, secret_data, approximation):
    row_errors = np.zeros((data.shape[0],))
    for i in range(data.shape[0]):
        row_errors[i] = np.where(secret_data[i,:] != 0 , np.square(approximation[i,:] - secret_data[i,:]), 0).sum()
    return row_errors.sum() / (training_data!=0).sum()


# load data from file
print('Loading data file...')
raw_data, nb_ratings = load_data(DATA_FILE)
if INJECT_TEST_DATA:
    raw_data = np.array([
        [0, 0, 5, 4, 0, 0],
        [0, 2, 0, 0, 0, 1],
        [6, 0, 0, 0, 8, 0],
        [0, 2, 1, 0, 0, 0],
        ])
    nb_ratings = (raw_data!=0).sum()


# prepare training set
if DO_LOCAL_VALIDATION:
    print('Building validation set...')
    training_data, secret_data = build_validation_set(raw_data)
else:
    training_data = raw_data


# run prediction
print('Running Model {}...'.format(MODEL))
if MODEL == 'average':
    predictions = average_prediction(training_data)
elif MODEL == 'SVD':
    predictions = svd_prediction(average_prediction(training_data), K=12)
elif MODEL == 'HYPER_SVD': 
    # try SVD with different K values
    Ks = list(range(4, 20))
    scores = np.zeros((len(Ks), 100))
    for epoch in range(100):
        # reshuffle data in every epoch
        non_zero_indices = list(zip(*np.nonzero(raw_data)))
        shuffled_indices = np.random.shuffle(non_zero_indices)
        secret_set_indices = non_zero_indices[:nb_ratings//2]
        known_set_indices = non_zero_indices[nb_ratings//2:]
        training_data = np.array(raw_data)
        training_data[list(zip(*secret_set_indices))] = 0
        secret_data = np.array(raw_data)
        secret_data[list(zip(*known_set_indices))] = 0
        
        for i, K in enumerate(Ks):
            print('Epoch {}: predicting using K={}...'.format(epoch, K))
            scores[i][epoch] = validate(raw_data, training_data, secret_data, svd_prediction(average_prediction(training_data), K=K))
    pickle.dump(scores, open('scores.pkl', 'wb'))
    plt.plot(Ks,np.average(scores, axis=1))
elif MODEL == 'SGD':
    predictions = sgd_prediction(training_data)


# validate locally or export prediction for submission
if MODEL != 'HYPER_SVD':
    # either do local score computation or write submission
    if DO_LOCAL_VALIDATION:
        print('Running local validation (not writing submission file)...')
        score = validate(raw_data, training_data, secret_data, predictions)
        print('Score = {}'.format(score))
    else:
        print('Writing to file...')     
        filename = TARGET_FOLDER + '/submission_{}_{}.csv'.format(USERNAME, time.strftime('%c').replace(':','-')[4:-5])
        write_data(filename, predictions)       
