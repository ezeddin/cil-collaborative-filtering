# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse
import time
from local_vars import USERNAME
import pickle
import matplotlib.pyplot as plt
import math
#%matplotlib inline

DATA_FILE = 'data/data_train.csv'
SUBMISSION_EXAMPLE = 'data/sampleSubmission.csv'
TARGET_FOLDER = 'submissions'
SUBMISSION_FORMAT='r{}_c{},{:.3f}\n'

NB_USERS = 10000
NB_ITEMS = 1000

DO_LOCAL_VALIDATION = True
VALIDATION_AVERAGING = 2
MODEL = 'SVD'
HYPER_PARAM = [5,10,15,20]
INJECT_TEST_DATA = False
ROUND = 15 
POST_PROCESS = True

def load_data(filename):
    print("Loading data...")
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

    if INJECT_TEST_DATA:
        data = np.array([
            [0, 0, 5, 4, 0, 0],
            [0, 2, 0, 0, 0, 1],
            [6, 0, 0, 0, 8, 0],
            [0, 2, 1, 0, 0, 0],
            ])
        nb_ratings = (raw_data!=0).sum()

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

def print_stats(matrix):
    print("The matrix has the following values:")
    print("  > Values range from {} to {}".format(np.min(matrix),np.max(matrix)))
    print("  > Average value is {}".format(np.average(matrix)))

def write_data(filename, matrix):
    print("Writing to file...")     
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
                
                rating = np.round(matrix[item,user], ROUND)
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

def post_process(predictions):
    predictions[predictions > 5.0] = 5.0
    predictions[predictions < 1.0] = 1.0
    
def run_model(training_data, param1):
    if MODEL == 'average':
        predictions = average_prediction(training_data)
    elif MODEL == 'SVD':
        predictions = svd_prediction(average_prediction(training_data), K=param1)
    elif MODEL == 'SGD':
        predictions = sgd_prediction(training_data)
    if POST_PROCESS:
        post_process(predictions)
    return predictions

def validate(training_data, secret_data, approximation):
    row_errors = np.zeros((training_data.shape[0],))
    for i in range(training_data.shape[0]):
        row_errors[i] = np.where(secret_data[i,:] != 0 , np.square(approximation[i,:] - secret_data[i,:]), 0).sum()
    return math.sqrt(row_errors.sum() / (secret_data!=0).sum())


# load data from file
raw_data, nb_ratings = load_data(DATA_FILE)

# run prediction
print('Running {}...'.format(MODEL))
if DO_LOCAL_VALIDATION:
    hyper_parameters = HYPER_PARAM if type(HYPER_PARAM) == list else [HYPER_PARAM]
    scores = []
    for param in hyper_parameters:
        avg_scores = []
        for avg_epoch in range(VALIDATION_AVERAGING):
            print('  > shuffling data for averaging epoch: {}'.format(avg_epoch))
            training_data, secret_data = build_validation_set(raw_data)
            print('  > running model...'.format(avg_epoch))
            avg_scores.append(validate(training_data, secret_data, run_model(training_data, param)))
        scores.append([param, np.average(avg_scores)])
        print('    > Score = {} for param='.format(scores[-1][1]), param)
    if len(scores) > 1:
        npscore = np.array(scores)
        pickle.dump(npscore, open('data/scores_{}.pkl'.format(time.strftime('%c').replace(':','-')[4:-5]), 'wb'))
        plt.plot(npscore[:,0], npscore[:,1])
        plt.xlabel('param')
        plt.ylabel('score')
else:
    training_data = raw_data
    assert type(HYPER_PARAM)!=list, "We want to export a submission! Hyperparameter can't be a list!"
    predictions = run_model(raw_data, HYPER_PARAM)
    print_stats(predictions) 
    filename = TARGET_FOLDER + '/submission_{}_{}.csv'.format(USERNAME, time.strftime('%c').replace(':','-')[4:-5])
    write_data(filename, predictions)
