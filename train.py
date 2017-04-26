# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse
import time
from local_vars import USERNAME

DATA_FILE = 'data/data_train.csv'
SUBMISSION_EXAMPLE = 'data/sampleSubmission.csv'
TARGET_FOLDER = 'submissions'
SUBMISSION_FORMAT='r{}_c{},{}\n'

NB_USERS = 10000
NB_ITEMS = 1000

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
    return values, rows, cols

def write_data(filename, matrix):
    print("Writing to file...")     
    with open(filename, 'w') as write_file:
        write_file.write("Id,Prediction\n")
        with open(SUBMISSION_EXAMPLE, 'r') as example_file:
            lines = example_file.readlines()
            
            for i, l in enumerate(lines[1:]):
                s = l.split(',')
                #values[i] = float(s[1].strip())
                c = s[0].split('_')
                user = int(c[0][1:]) - 1 # 0-based indexing
                item = int(c[1][1:]) - 1 # 0-based indexing
                
                write_file.write(SUBMISSION_FORMAT.format(user+1, item+1,matrix[item,user]))

def print_stats(matrix):
    print("The matrix has the following values:")
    print("Values range from {} to {}".format(np.min(matrix),np.max(matrix)))
    print("Average value is {}".format(np.average(matrix)))
    
values, rows, cols = load_data(DATA_FILE)

data = scipy.sparse.csr_matrix((values, (rows, cols)), shape=(NB_ITEMS, NB_USERS), dtype='float')
nb_ratings = (data!=0).sum()

print('Dataset has {} non zero values'.format(nb_ratings))
print('average rating : {}'.format( data.sum() / nb_ratings))

data = data.todense()

predictions = data

average_for_item = np.true_divide(predictions.sum(1), (predictions!=0).sum(1))
for i in range(predictions.shape[0]):
    predictions[i,:] = np.where(predictions[i,:] == 0, average_for_item[i], predictions[i,:])
                
K = 10
print("Computing SVD...")
U, S, VT = np.linalg.svd(predictions, full_matrices=True)

U2 = U[:,:K]
S2 = S[:K]
VT2 = VT[:K, :]

reduced_data = U2.dot(np.diag(S2)).dot(VT2)
      
print_stats(reduced_data) 
filename = TARGET_FOLDER + "/submission_{}_{}.csv".format(USERNAME, time.strftime("%c").replace(":","-")[4:-5])
write_data(filename, reduced_data)
