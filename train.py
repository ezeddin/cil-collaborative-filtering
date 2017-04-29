# -*- coding: utf-8 -*-

import argparse
import numpy as np
import scipy.sparse
import time
import datetime
from local_vars import USERNAME
import pickle
import matplotlib.pyplot as plt
import math
from helpers import Logger
import random 

#%matplotlib inline

DATA_FILE = 'data/data_train.csv'
SUBMISSION_EXAMPLE = 'data/sampleSubmission.csv'
TARGET_FOLDER = 'submissions'
ROUND = 5 
SUBMISSION_FORMAT='r{{}}_c{{}},{{:.{}f}}\n'.format(ROUND)

NB_USERS = 10000
NB_ITEMS = 1000

SGD_ITER = 60000000
INJECT_TEST_DATA = False
args = None


old_settings = np.seterr(all='raise')

def main(arguments, matrix=None):
    global args
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--submission', type=bool, default=False,
                        help='Omit local validation, just export submission file.')
    parser.add_argument('--model', type=str, default='SGD',
                        help='Prediction algorithm: average, SVD, SVD2, SGD')
    parser.add_argument('--cv_splits', type=int, default=8,
                        help='Data splits for cross validation')
    parser.add_argument('--score_averaging', type=int, default=1,
                        help='On how many of the splits should be tested?')
    parser.add_argument('--param', type=str, default="12",
                        help='Hyper parameter, can also be a list')
    parser.add_argument('--L', type=float, default=0.1,
                        help='Hyper parameter for SGD')
    parser.add_argument('--lr_factor', type=float, default=1.0,
                        help='Multiplier for the learning rate.')
    parser.add_argument('--postproc', type=bool, default=True,
                        help='Do post procession like range cropping')
    parser.add_argument('--v', type=int, default=2,
                        help='Verbosity of sgd: 0 for nothing, 1 for basic messages, 2 for steps')
    parser.add_argument('--n_messages', type=int, default=20,
                        help='The number of messages to print for the sgd. Only relevant when --v==2')

    parser.add_argument('--external_matrix', type=bool, default=False,
                        help='In a multiprocessing environment: get matrices from external arguments')
    args = parser.parse_args(arguments)
    args.param = eval(args.param)
    args.param = args.param if type(args.param) == list else [args.param]

    if not args.external_matrix:
        # load data from file
        return train(load_data(DATA_FILE))
    else:
        # data is given in argument to main()
        return train(matrix)

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
    nb_ratings = data.getnnz()

    if INJECT_TEST_DATA: 
        data = np.array([
            [0, 0, 5, 4, 0, 0],
            [0, 2, 0, 0, 0, 1],
            [6, 0, 0, 0, 8, 0],
            [0, 2, 1, 0, 0, 0],
            ])
        nb_ratings = (data!=0).sum()

    print('Dataset has {} non zero values'.format(nb_ratings))
    print('average rating : {}'.format( data.sum() / nb_ratings))
    return np.asarray(data.todense())


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
                
    
def print_stats(matrix):
    print("The matrix has the following values:")
    print("  > Values range from {} to {}".format(np.min(matrix),np.max(matrix)))
    print("  > Average value is {}".format(np.average(matrix)))
    

def split_randomly(raw_data, n_splits=8):
    """
        randomly splits the non-zero indices of the raw_data matrix into n parts. 
        a list of length n is returned, each element being the indices of the i-th split
    """
    non_zero_indices = list(zip(*np.nonzero(raw_data)))
    np.random.shuffle(non_zero_indices) 
    size = len(non_zero_indices) // n_splits
    assert(size * n_splits == len(non_zero_indices)), "n chosen for cross validation does not evenly split data. Choose 4, 7, 8, 14, 28, 56"
    
    return [non_zero_indices[(a*size): ((a+1)*size)] for a in range(n_splits)]


def build_train_and_test(raw_data, index_splits, use_as_test):
    """
    raw_data : the matrix to split into parts
    index_splits : list of list of indexes as returned by slpit_randomly function
    use_as_test : the split (int) to leave out as test set
    
    returns : a training matrix and a test matrix
    """
    train_data = np.array(raw_data)
    train_data[list(zip(*index_splits[use_as_test]))] = 0 #TODO : optimize this line
    
    test_data = np.array(raw_data)
    for i in range(len(index_splits)):
        if (i == use_as_test):
            continue
        test_data[list(zip(*index_splits[i]))] = 0 #TODO : optimize this line
    assert(np.array_equal(train_data+test_data, raw_data))
    return train_data, test_data


def averaging_fill_up(matrix):
    """
        fill up the sparse matrix row by row with the average value of this row
    """
    average_for_item = np.true_divide(matrix.sum(1), (matrix!=0).sum(1))
    averaged = np.array(matrix, np.float64)
    for i in range(matrix.shape[0]):
        averaged[i,:] = np.where(averaged[i,:] == 0, average_for_item[i], averaged[i,:])
    return averaged


def sampling_distribution_fill_up(matrix):
    """
        fill up the sparse matrix row by row by sampling from the empirical distribution
        of the rating in this row.
    """
    filled = np.array(matrix)
    for i in range(matrix.shape[0]):
        nonzeros_per_row = np.squeeze(matrix[i, np.where(matrix[i,:] != 0.0)[0]])
        for j in range(matrix.shape[1]):
            if filled[i,j] == 0:
                filled[i,j] = np.random.choice(nonzeros_per_row, replace=True)
    return filled


def svd_prediction(matrix, K=15):
    """
        computes SVD from filled up data matrix (not sparse anymore)
        
        K: number of singular values to keep (number of 'principle components')
    """
    U, S, VT = np.linalg.svd(matrix, full_matrices=True)
    U2 = U[:,:K]
    S2 = S[:K]
    VT2 = VT[:K, :]
    return U2.dot(np.diag(S2)).dot(VT2)


def sgd_prediction(matrix, test_data, K, verbose, L=0.1):
    """
        matrix is the training dataset with nonzero entries only where ratings are given
        
        verbose = 0 for no logging
                  1 for inital messages
                  2 for steps
    """
    
    print_every = SGD_ITER / args.n_messages
    U = np.random.rand(matrix.shape[0],K)
    V = np.random.rand(matrix.shape[1],K)
    
    
    non_zero_indices = list(zip(*np.nonzero(matrix)))
    if verbose > 0 :
        print("      SGD: sgd_prediction called. K = {}, L = {}".format(K, L))
        print("      SGD: There are {} nonzero indices in total.".format(len(non_zero_indices)))
    
    lr = sgd_learning_rate(0,0)
    start_time = datetime.datetime.now()
    for t in range(SGD_ITER):
        if t % 1000000 == 0:
            lr = sgd_learning_rate(t, lr)
        d,n = random.choice(non_zero_indices)
            
        
        #TODO : if convergence is slow, we could use a bigger batch size (update more indexes at once)
        U_d = U[d,:]
        V_n = V[n,:]
        delta = matrix[d,n] - U_d.dot(V_n)

        try:
            new_U_d = U_d + lr * (delta * V_n - L*U_d)
            new_V_n = V_n + lr * (delta * U_d - L*V_n)
        except FloatingPointError:
            print ("WARNING: FLOATING POINT ERROR CAUGHT!")
        else:
            U[d,:] = new_U_d
            V[n,:] = new_V_n
        
        if verbose == 2 and t % print_every == 0:
            score = validate(matrix, U.dot(V.T))
            test_score = validate(test_data, U.dot(V.T)) if test_data is not None else -1
            print("      SGD : step {:8d}  ({:2d} % done!). fit = {:.4f}, test_fit={:.4f}, lr={:.5f}".format(t+1, int(100 * (t+1) /SGD_ITER), score, test_score, lr))
        if t == 500000:
            t_after_100 = datetime.datetime.now() - start_time;
            if args.submission:
                multi_runs = 1
            else:
                multi_runs = len(args.param)*args.score_averaging
            duration = t_after_100/500000*SGD_ITER*multi_runs
            end = datetime.datetime.now() + duration
            print("    Expected duration: {}, ending at time {}".format(str(duration).split('.')[0], str(end).split('.')[0]))        
    return U.dot(V.T)



def sgd_learning_rate(t, current):
    result = 0
    percent_done = t / SGD_ITER
    if  (t < 2/6): 
        result =  0.03
    elif(t < 3/6): 
        result =  0.01
    elif(t < 4/6): 
        result =  0.002
    elif(t < 5/6): 
        result =  0.0005
    elif(t < 5.5/6):
        result =  0.0001
    else:
        result =  0.00002
    return result * args.lr_factor


def post_process(predictions):
    predictions[predictions > 5.0] = 5.0
    predictions[predictions < 1.0] = 1.0


def run_model(training_data, test_data, param1):
    if args.model == 'average':
        predictions = averaging_fill_up(training_data)
    elif args.model == 'SVD':
        predictions = svd_prediction(averaging_fill_up(training_data), K=param1)
    elif args.model == 'SVD2':
        predictions = svd_prediction(sampling_distribution_fill_up(training_data), K=param1)
    elif args.model == 'SGD':
        predictions = sgd_prediction(training_data, test_data, K=param1, verbose=args.v, L=args.L)
    if args.postproc:
        post_process(predictions)
    return predictions


def validate(secret_data, approximation):
    """
        calculate the score for approximation when predicting secret_data, using the same formula as on kaggle
    """
    error_sum = np.where(secret_data!=0, np.square(approximation-secret_data),0).sum()
    return math.sqrt(error_sum / (secret_data!=0).sum())


def train(raw_data):
    """
        Main routine that loads data and trains the model. At the end, it either
        exports a submission file or it exports the scores from the local cross
        validation as a pickle file.
    """
    print('Running {}...'.format(args.model))
    if not args.submission:
        scores = []
        print("creating {} splits for Cross-Validation!".format(args.cv_splits))
        splits = split_randomly(raw_data, args.cv_splits)        
        for param in args.param:
            print("Testing with hyperparameter {}".format(param))
            avg_scores = []
            for i in range(args.cv_splits):
                if i >= args.score_averaging:
                    continue
                training_data, test_data = build_train_and_test(raw_data, splits, i)
                print('    running model when leaving out split {}'.format(i))
                avg_scores.append(validate(test_data, run_model(training_data,test_data,  param)))
                print('    got score : {}'.format(avg_scores[-1]))

            scores.append([param, np.average(avg_scores)])
            print('  score = {} for param='.format(scores[-1][1]), param)

        print('Saving final score in data/scores_<timestamp>.pkl')
        npscore = np.array(scores)
        if len(args.param) > 1:
            K_param = str(args.param)
        else:
            K_param = str(args.param[0])
        score_filename = 'data/scores_{}_{}_{:.3}.pkl'.format(time.strftime('%c').replace(':','-')[4:-5], K_param, args.L)
        pickle.dump(npscore, open(score_filename, 'wb'))
        if len(scores) > 1:
            try:
                plt.plot(npscore[:,0], npscore[:,1])
                plt.xlabel('param')
                plt.ylabel('score')
            except:
                print('Plotting not working.')
        return scores
    else:
        training_data = raw_data
        assert len(args.param) == 1, "We want to export a submission! Hyperparameter can't be a list!"
        predictions = run_model(raw_data, None, args.param[0])
        print_stats(predictions)
        filename = TARGET_FOLDER + '/submission_{}_{}_{}_{}.csv'.format(USERNAME, time.strftime('%c').replace(':','-')[4:-5], args.param, args.L)
        write_data(filename, predictions)
        return predictions


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
