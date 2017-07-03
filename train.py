# -*- coding: utf-8 -*-

import argparse
import numpy as np
import scipy.sparse
import time
from local_vars import USERNAME
import pickle
import matplotlib.pyplot as plt
import math
import random 
from keras.models import Sequential
from keras.layers import Dense,  Dropout
import keras.callbacks
import datetime

_bool = lambda s: s.lower() in ['true', 't', 'yes', '1']

#%matplotlib inline

DATA_FILE = 'data/data_train.csv'
SUBMISSION_EXAMPLE = 'data/sampleSubmission.csv'
TARGET_FOLDER = 'submissions'
ROUND = 5 
SUBMISSION_FORMAT='r{{}}_c{{}},{{:.{}f}}\n'.format(ROUND)

NB_USERS = 10000
NB_ITEMS = 1000

SGD_ITER = 60000000

args = None

np.random.seed(0)
random.seed(0)

old_settings = np.seterr(all='raise')

def main(arguments, matrix=None):
    global args
    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--submission', type=bool, default=True,
                        help='Omit local cross-validation, just train the model and export submission file. ')
    parser.add_argument('--model', type=str, default='SGD+',
                        help='Prediction algorithm: average, SVD, SGD, SGD+, SGDnn')
    parser.add_argument('--cv_splits', type=int, default=14,
                        help='Number of data splits for cross validation')
    parser.add_argument('--score_averaging', type=int, default=1,
                        help='How many splits should be used as test data')
    parser.add_argument('--K', type=str, default="12",
                        help='First hyperparameter : Latent feature dimension size. int or list')
    parser.add_argument('--L', type=float, default=0.083,
                        help='Second hyperparameter : Regularizer for SGD.')
    parser.add_argument('--L2', type=float, default=0.04,
                        help='Third hyperparameter : Bias regularizer for SGD+')
    parser.add_argument('--lr_factor', type=float, default=3.0,
                        help='Multiplier for the learning rate.')
    parser.add_argument('--v', type=int, default=2,
                        help='Verbosity of sgd algorithm: 0 for none, 1 for basic messages, 2 for steps')
    parser.add_argument('--n_messages', type=int, default=20,
                        help='The number of messages to print for the sgd. Only relevant when --v==2')
    parser.add_argument('--external_matrix', type=bool, default=False,
                        help='In a multiprocessing environment: get matrices from external arguments')
    parser.add_argument('--model_path', type=str, default=None,
                    help='load matrices from external arguments (use to skip SGD+ when using a neural network postprocessing)')
    parser.add_argument('--save_model', type=bool, default=False,
                    help='save all matrices')
    parser.add_argument('--dropout', type=float, default=0.6,
                        help='Dropout value to use in neural network postprocessing.')
    parser.add_argument('--early_stopping', type=_bool, default=False,
                        help='Observes the loss and stops early if it doesn\'t decrease.')
    args = parser.parse_args(arguments)
    args.K = eval(args.K)
    args.K = args.K if type(args.K) == list else [args.K]

    if args.external_matrix:
        # data is given in argument to main()
        return train(matrix)
    else:
        # load data from file
        return train(load_data(DATA_FILE))

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
    index_splits : list of list of indexes as returned by split_randomly function
    use_as_test : the split (int) to leave out as test set
    
    returns : a training matrix and a test matrix
    """
    train_data = np.array(raw_data)
    train_data[list(zip(*index_splits[use_as_test]))] = 0 
    
    test_data = np.array(raw_data)
    for i in range(len(index_splits)):
        if (i == use_as_test):
            continue
        test_data[list(zip(*index_splits[i]))] = 0
    assert(np.array_equal(train_data+test_data, raw_data))
    return train_data, test_data

def averaging_prediction(matrix):
    """
        A very naive model.
        fill up the sparse matrix row by row with the average value of this row
    """
    average_for_item = np.true_divide(matrix.sum(1), (matrix!=0).sum(1))
    averaged = np.array(matrix, np.float64)
    for i in range(matrix.shape[0]):
        averaged[i,:] = np.where(averaged[i,:] == 0, average_for_item[i], averaged[i,:])
    return averaged

def svd_prediction(matrix, K=15):
    """
        computes the SVD from filled up data matrix and returns the prediction for non-negative values.
        
        matrix : The matrix (dense) for which to calculate the singular value decomposition
        K: number of singular values to keep (number of 'principle components')
        
    """
    U, S, VT = np.linalg.svd(matrix, full_matrices=True)
    U2 = U[:,:K]
    S2 = S[:K]
    VT2 = VT[:K, :]
    return U2.dot(np.diag(S2)).dot(VT2)


def sgd_prediction_nobias(matrix, test_data, K, L, verbose):
    """
        A simple Stochastic gradient descent predictor. Does not use biases.
        
        matrix is the training dataset with nonzero entries only where ratings are given
        
        test_data is used to calculate the test scores.
        
        verbose = 0 for no logging
                  1 for inital messages
                  2 for steps
                  
    """
    non_zero_indices = list(zip(*np.nonzero(matrix)))
    global_mean = matrix.sum() / len(non_zero_indices)
    
    print_every = SGD_ITER / args.n_messages
    
    U = np.random.uniform(0, 0.05, (matrix.shape[0], K))
    V = np.random.uniform(0, 0.05, (matrix.shape[1], K))
    
    
    if verbose > 0 :
        print("      SGD: sgd_prediction called. No biases, K={}, L={}, lr_factor={}".format(K, L, args.lr_factor))
        print("      SGD: There are {} nonzero indices in total.".format(len(non_zero_indices)))
        print("      SGD: global mean is {}".format(global_mean))
        
    lr = sgd_learning_rate(0)
    for t in range(SGD_ITER):
        if t % 1000000 == 0: #update learning rate
            lr = sgd_learning_rate(t)
        d,n = random.choice(non_zero_indices)
                        
        U_d = U[d,:]
        V_n = V[n,:]

        guess = U_d.dot(V_n)            
        delta = matrix[d,n] - guess

        try:
            new_U_d = U_d + lr * (delta * V_n - L*U_d)
            new_V_n = V_n + lr * (delta * U_d - L*V_n)
        except FloatingPointError:
            print("WARNING : FloatingPointError caught! Iteration skipped!")
            continue
        else:
            U[d,:] = new_U_d
            V[n,:] = new_V_n
                
        if verbose == 2 and t % print_every == 0:
            score = validate(matrix, U.dot(V.T))
            test_score = validate(test_data, U.dot(V.T)) if test_data is not None else -1
            print("      SGD : step {:8d}  ({:2d}% done). fit = {:.4f}, test_fit={:.4f}, learning_rate={:.5f}".format(t+1, int(100 * (t+1) /SGD_ITER), score, test_score, lr))
    return U.dot(V.T)
    
    
    
    
def sgd_prediction(matrix, test_data, K,  L, L2, verbose, postprocess=False):
    """
        stochastic gradient descent predictor.
        It is the model that leads to the best scores.
        
        matrix is the training dataset with nonzero entries only where ratings are given
        
        K : the number of features to use
        L : regularizer for the 
        L2 : regularizer for 
        
        postprocess : if True, the neural network postprocessing will be used
        verbose = 0 for no logging
                  1 for inital messages
                  2 for steps
                  

    """
    
    if args.model_path:
        U, V, biasU, biasV = pickle.load(open(args.model_path, 'rb'))
    else:
        non_zero_indices = list(zip(*np.nonzero(matrix)))
        global_mean = matrix.sum() / len(non_zero_indices)
        
        print_every = SGD_ITER / args.n_messages
        
        U = np.random.uniform(0, 0.05, (matrix.shape[0], K))
        V = np.random.uniform(0, 0.05, (matrix.shape[1], K))
    
        biasU = np.zeros(matrix.shape[0])
        biasV = np.zeros(matrix.shape[1])
            
        
        if verbose > 0 :
            print("      SGD: sgd_prediction called. Using Biases, K={}, L={}, L2={}, lr_factor={}".format(K, L, L2, args.lr_factor))
            print("      SGD: There are {} nonzero indices in total.".format(len(non_zero_indices)))
            print("      SGD: global mean is {}".format(global_mean))
            
        lr = sgd_learning_rate(0)
        for t in range(SGD_ITER):
            if t % 1000000 == 0:
                lr = sgd_learning_rate(t)
            d,n = random.choice(non_zero_indices)
                
            
            U_d = U[d,:]
            V_n = V[n,:]

            biasU_d = biasU[d]
            biasV_n = biasV[n]

            guess = U_d.dot(V_n) + biasU_d + biasV_n
        
            delta = matrix[d,n] - guess

            try:
                new_U_d = U_d + lr * (delta * V_n - L*U_d)
                new_V_n = V_n + lr * (delta * U_d - L*V_n)

                new_biasU_d = biasU_d + lr * ( delta - L2*(biasU_d + biasV_n - global_mean))
                new_biasV_n = biasV_n + lr * ( delta - L2*(biasV_n + biasU_d - global_mean))
                
            except FloatingPointError:
                print("WARNING : FloatingPointError caught! Iteration skipped!")
                continue
            else:
                U[d,:] = new_U_d
                V[n,:] = new_V_n
                
                biasU[d] = new_biasU_d
                biasV[n] = new_biasV_n
                
            if verbose == 2 and t % print_every == 0:
                score = validate(matrix, U.dot(V.T) + biasU.reshape(-1,1) + biasV)
                test_score = validate(test_data, U.dot(V.T) + biasU.reshape(-1,1) + biasV) if test_data is not None else -1
                print("      SGD : step {:8d}  ({:2d}% done). fit = {:.4f}, test_fit={:.4f}, learning_rate={:.5f}".format(t+1, int(100 * (t+1) /SGD_ITER), score, test_score, lr))

    if args.save_model:
        filename = 'save/mode_SGD_{}_{}_{:.4}_{:.4}.pkl'.format(time.strftime('%c').replace(':','-')[4:-5], K, L, L2)
        pickle.dump([U, V,  biasU, biasV], open(filename, 'wb'))
    
    if postprocess:
        return retrain_U(matrix, test_data, V, biasV)
    else:
        return U.dot(V.T) + biasU.reshape(-1,1) + biasV 


def retrain_U(matrix, test_data, V, biasV):
    """
        Postprocessing used in SGDnn model. 
        Recomputes the U matrix as well as biasU that we discarded.
        
        matrix : The original matric
        V : The movie features that were computed by the SGD.
        biasV : The movie biases that were computed by the SGD.
        
        Returns : The full recalculated matrix.
    """
    K = V.shape[1]

    pred_matrix = np.zeros(matrix.shape)
    time_start = datetime.datetime.now()
    for i in range(matrix.shape[0]): #retrain for each user
        non_zero_indices = np.where(matrix[i] != 0)[0]
        input_data = V[non_zero_indices]
        output_data = (matrix[i]-biasV)[non_zero_indices]
        
    
        model = Sequential()    
        model.add(Dropout(args.dropout, input_shape=[K]))
        model.add(Dense(1, init='uniform', activation='linear'))
        model.compile(loss='mse', optimizer='sgd')

        if args.early_stopping:
            early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss', verbose=0, mode='auto', patience=4)
            model.fit(input_data, output_data, validation_split=0.1, verbose=2, nb_epoch=300, callbacks=[early_stopping])
        else:
            model.fit(input_data, output_data, verbose=0, nb_epoch=300)
            
        pred_input_data = V
        pred_output_data = model.predict(pred_input_data, verbose=2)
        pred_matrix[i] = (pred_output_data.T)[0]
        duration = (datetime.datetime.now()-time_start)*(matrix.shape[0]/(i+1) - 1)
        duration_str = 'h '.join(str(duration).split('.')[0].split(':')[0:2]) + 'm'
        print("user {}, expected duration: {}".format(i, duration_str))
    return pred_matrix + biasV


def sgd_learning_rate(t):
    """
    learning rate used in sgd.
    
    t : the step number
    """
    result = 0
    done = t / SGD_ITER
    if   done < 2/6: 
        result =  0.03
    elif done < 3/6: 
        result =  0.01
    elif done < 4/6: 
        result =  0.002
    elif done < 5/6: 
        result =  0.0005
    elif done < 5.5/6:
        result =  0.0001
    else:
        result =  0.00002
    return result * args.lr_factor

def clip_values(predictions):
    predictions[predictions > 5.0] = 5.0
    predictions[predictions < 1.0] = 1.0
    return predictions

def validate(secret_data, approximation):
    """
        calculate the score for approximation when predicting secret_data, using the same formula as on kaggle
    """
    error_sum = np.where(secret_data!=0, np.square(approximation-secret_data),0).sum()
    return math.sqrt(error_sum / (secret_data!=0).sum())


def run_model(training_data, test_data, K):
    if args.model == 'average':
        predictions = averaging_prediction(training_data)
    elif args.model == 'SVD':
        predictions = svd_prediction(averaging_prediction(training_data), K=K)
    elif args.model == 'SGD':
        predictions = sgd_prediction_nobias(training_data, test_data, K=K, L=args.L, L2=args.L2, verbose=args.v)
    elif args.model == 'SGD+':
        predictions = sgd_prediction(training_data, test_data, K=K,  L=args.L, L2=args.L2, verbose=args.v)
    elif args.model == 'SGDnn':
        predictions = sgd_prediction(training_data, test_data, K=K,  L=args.L, L2=args.L2, verbose=args.v, postprocess=True)
    else:
        assert 'Model not supported'
    return clip_values(predictions)


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
        for K in args.K:
            print("Testing with K = {}".format(K))
            avg_scores = []
            for i in range(args.cv_splits):
                if i >= args.score_averaging:
                    continue
                training_data, test_data = build_train_and_test(raw_data, splits, i)
                print('    running model when leaving out split {}'.format(i))
                avg_scores.append(validate(test_data, run_model(training_data, test_data, K)))
                print('    got score : {}'.format(avg_scores[-1]))

            scores.append([K, np.average(avg_scores)])
            print('  score = {} for K = {}'.format(scores[-1][1], K))

        print('Saving final score in data/scores_<timestamp>.pkl')
        npscore = np.array(scores)
        if len(args.K) > 1:
            K_str = str(args.K)
        else:
            K_str = str(args.K[0])
        score_filename = 'data/scores_{}_{}_{:.4}_{:.4}.pkl'.format(time.strftime('%c').replace(':','-')[4:-5], K_str, args.L, args.L2)
        pickle.dump(npscore, open(score_filename, 'wb'))
        if len(scores) > 1:
            try:
                plt.plot(npscore[:,0], npscore[:,1])
                plt.xlabel('K')
                plt.ylabel('score')
            except:
                print('Plotting not working.')
        return scores
    else:
        assert len(args.K) == 1, "We want to export a submission! Hyperparameter can't be a list!"
        predictions = run_model(raw_data, None, args.K[0])
        filename = TARGET_FOLDER + '/submission_{}_{}_{}_{}_d{}.csv'.format(USERNAME, time.strftime('%c').replace(':','-')[4:-5], args.K, args.L, args.dropout)
        write_data(filename, predictions)
        return predictions


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
