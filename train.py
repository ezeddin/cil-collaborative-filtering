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
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras.callbacks
import os
from keras import regularizers

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
    parser.add_argument('--submission', type=bool, default=False,
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
    parser.add_argument('--subtract_mean', type=bool, default=False,
                        help='Subtracts the mean of the data matrix in preprocessing')
    parser.add_argument('--external_matrix', type=bool, default=False,
                        help='In a multiprocessing environment: get matrices from external arguments')
    parser.add_argument('--model_path', type=str, default=None,
                    help='load all matrices from external arguments')
    parser.add_argument('--save_model', type=bool, default=False,
                    help='save all matrices')
    parser.add_argument('--uv_init_mean', type=float, default=None, #TODO : Delete
                        help='Mean value of random initialization of U and V matrices (if None, depends on K)')
    parser.add_argument('--uv_init_std', type=float, default=0.1, #TODO : Delete
                        help='Standard deviation of random initialization of U and V matrices')
    parser.add_argument('--bias_init_mean', type=float, default=None, #TODO : Delete
                        help='Mean value of random initialization of bias vectors (if None, initialization with row and column averages)')
    parser.add_argument('--bias_init_std', type=float, default=0.0, #TODO : Delete
                        help='Standard deviation of random initialization of bias vectors')
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

def averaging_prediction(matrix):
    """
        fill up the sparse matrix row by row with the average value of this row
    """
    average_for_item = np.true_divide(matrix.sum(1), (matrix!=0).sum(1))
    averaged = np.array(matrix, np.float64)
    for i in range(matrix.shape[0]):
        averaged[i,:] = np.where(averaged[i,:] == 0, average_for_item[i], averaged[i,:])
    return averaged

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

def retrain_U(matrix, test_data, V):
    K = V.shape[1]
    configs = [
        [(K,"relu")],
        [(2*K, "relu"), (K,"relu")],
        [(2*K, "relu"), (K,"relu"),(K/2, "relu")]
    ]

    scores = []
    #for config in configs:
    pred_matrix = np.copy(matrix)
    for i in range(matrix.shape[0]): #retrain for each user

        non_zero_indices = np.where(matrix[i] != 0)[0]
        zero_indices = np.where(matrix[i] == 0)[0]
        input_data = V[non_zero_indices]
        output_data = matrix[i][non_zero_indices]
        
        """
        non_zero_indices_val = np.where(test_data[i] != 0)[0]
        zero_indices_val = np.where(test_data[i] == 0)[0]
        input_data_val = V[non_zero_indices_val]
        output_data_val = test_data[i][non_zero_indices_val]
        """
        model = Sequential()
        #l_units, l_activtor = config[0]
        #model.add(Dense(units=l_units, input_dim=K, kernel_initializer='normal', activation=l_activtor))
        #for l_units, l_activtor in config[1:]:
        #    model.add(Dense(units=l_units, kernel_initializer='normal', activation=l_activtor))
            # model.add(Dropout(0.5))
        # model.add(Dense(units=1, kernel_initializer='normal'))
        ## Compile model
        # model.compile(loss='mean_squared_error', optimizer='adam')

        model.add(Dropout(0.6, input_shape=[K]))
        model.add(Dense(1, init='uniform', activation='linear'))
        model.compile(loss='mse', optimizer='sgd')

        early_stopping=keras.callbacks.EarlyStopping(monitor='val_loss', verbose=0, mode='auto', patience=3)
         
        model.fit(input_data, output_data, validation_split=0.1, verbose=2, nb_epoch=300, callbacks=[early_stopping])
        #model.fit(input_data, output_data, validation_data=(input_data_val, output_data_val),
        #          verbose=2,  nb_epoch=300, callbacks=[early_stopping])
        
        pred_input_data = V[zero_indices]
        pred_output_data = model.predict(pred_input_data, verbose=2)
        pred_matrix[i][zero_indices] = pred_output_data
        print("user {}, config {}".format(i, 1))
    #scores.append(validate(test_data,pred_matrix))
    #print("Scores obtained : " + str(scores))
    return pred_matrix

def sgd_prediction(matrix, test_data, K, verbose, L, L2, save_model=False, model_path=None, use_bias=True, use_nn=False):
    """
        matrix is the training dataset with nonzero entries only where ratings are given
        
        verbose = 0 for no logging
                  1 for inital messages
                  2 for steps
    """
    
    if model_path:
        U, V, optional_zero_mean, biasU, biasV = pickle.load(open(model_path, 'rb'))    
    else:
        non_zero_indices = list(zip(*np.nonzero(matrix)))
        global_mean = matrix.sum() / len(non_zero_indices)
        
        print_every = SGD_ITER / args.n_messages
        
        if use_bias:
            if args.bias_init_mean == None:
                biasU = np.zeros(matrix.shape[0])
                biasV = np.zeros(matrix.shape[1])
            else:
                biasU = np.random.normal(args.bias_init_mean, args.bias_init_std, matrix.shape[0])
                biasV = np.random.normal(args.bias_init_mean, args.bias_init_std, matrix.shape[1])

        if args.subtract_mean:
            optional_zero_mean = global_mean
        else:
            optional_zero_mean = 0.0
        mean = args.uv_init_mean if args.uv_init_mean else np.sqrt(global_mean/K)
        bound_u = mean - args.uv_init_std/2
        bound_l = mean + args.uv_init_std/2
        U = np.random.uniform(0, 0.02, (matrix.shape[0], K))
        V = np.random.uniform(0, 0.02, (matrix.shape[1], K))

        
        if verbose > 0 :
            print("      SGD: sgd_prediction called. biases={}, K={}, L={}, L2={}, lr_factor={}".format(use_bias, K, L, L2, args.lr_factor))
            print("      SGD: There are {} nonzero indices in total.".format(len(non_zero_indices)))
            print("      SGD: global mean is {}".format(global_mean))
        lr = sgd_learning_rate(0,0)
        start_time = datetime.datetime.now()
        for t in range(SGD_ITER):
            if t % 1000000 == 0:
                lr = sgd_learning_rate(t, lr)
            d,n = random.choice(non_zero_indices)
                
            
            #TODO : if convergence is slow, we could use a bigger batch size (update more indexes at once)
            U_d = U[d,:]
            V_n = V[n,:]

            guess = U_d.dot(V_n)
            if use_bias:
                biasU_d = biasU[d]
                biasV_n = biasV[n]
                guess += biasU_d + biasV_n
            
            delta = matrix[d,n] - guess - optional_zero_mean
            #delta = np.clip(delta, -1e5, 1e5)

            try:
                new_U_d = U_d + lr * (delta * V_n - L*U_d)
                new_V_n = V_n + lr * (delta * U_d - L*V_n)

                if use_bias:
                    new_biasU_d = biasU_d + lr * ( delta - L2*(biasU_d + biasV_n - global_mean + optional_zero_mean))
                    new_biasV_n = biasV_n + lr * ( delta - L2*(biasV_n + biasU_d - global_mean + optional_zero_mean))
            except FloatingPointError:
                print("WARNING : FloatingPointError caught! iteration skipped!")
                continue
            else:
                U[d,:] = new_U_d
                V[n,:] = new_V_n
                if use_bias : 
                    biasU[d] = new_biasU_d
                    biasV[n] = new_biasV_n
                    
            if verbose == 2 and t % print_every == 0:
                if use_bias:
                    score = validate(matrix, U.dot(V.T) + biasU.reshape(-1,1) + biasV + optional_zero_mean)
                    test_score = validate(test_data, U.dot(V.T) + biasU.reshape(-1,1) + biasV + optional_zero_mean) if test_data is not None else -1
                else:
                    score = validate(matrix, U.dot(V.T) + optional_zero_mean)
                    test_score = validate(test_data, U.dot(V.T) + optional_zero_mean) if test_data is not None else -1

                print("      SGD : step {:8d}  ({:2d}% done). fit = {:.4f}, test_fit={:.4f}, learning_rate={:.5f}".format(t+1, int(100 * (t+1) /SGD_ITER), score, test_score, lr))
            if t == 500000: 
                t_after_100 = datetime.datetime.now() - start_time;
                if args.submission:
                    multi_runs = 1
                else:
                    multi_runs = len(args.K)*args.score_averaging
                duration = t_after_100/500000*SGD_ITER*multi_runs
                end = datetime.datetime.now() + duration
                print("    Expected duration: {}, ending at time {}".format(str(duration).split('.')[0], str(end).split('.')[0]))

    
    
    if use_nn:
        if save_model:
            filename = 'save/mode_SGD_{}_{}_{:.4}_{:.4}.pkl'.format(time.strftime('%c').replace(':','-')[4:-5], K, L, L2)
            pickle.dump([U, V, optional_zero_mean, None, None], open(filename, 'wb'))
        return retrain_U(matrix, test_data, V)
    elif use_bias:
        if save_model:
            filename = 'save/mode_SGD+_{}_{}_{:.4}_{:.4}.pkl'.format(time.strftime('%c').replace(':','-')[4:-5], K, L, L2)
            pickle.dump([U, V, optional_zero_mean, biasU, biasV], open(filename, 'wb'))
        return U.dot(V.T) + biasU.reshape(-1,1) + biasV + optional_zero_mean
    else:
        if save_model:
            filename = 'save/mode_SGD_{}_{}_{:.4}_{:.4}.pkl'.format(time.strftime('%c').replace(':','-')[4:-5], K, L, L2)
            pickle.dump([U, V, optional_zero_mean, None, None], open(filename, 'wb'))
        return U.dot(V.T) + optional_zero_mean

def sgd_learning_rate(t, current):
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

def post_process(predictions):
    predictions[predictions > 5.0] = 5.0
    predictions[predictions < 1.0] = 1.0

def run_model(training_data, test_data, K):
    if args.model == 'average':
        predictions = averaging_prediction(training_data)
    elif args.model == 'SVD':
        predictions = svd_prediction(averaging_fill_up(training_data), K=K)
    elif args.model == 'SGD':
        predictions = sgd_prediction(training_data, test_data, K=K, verbose=args.v, L=args.L, L2=args.L2, save_model=args.save_model,model_path=args.model_path, use_bias=False)
    elif args.model == 'SGD+':
        predictions = sgd_prediction(training_data, test_data, K=K, verbose=args.v, L=args.L, L2=args.L2, save_model=args.save_model,model_path=args.model_path,)
    elif args.model == 'SGDnn':
        predictions = sgd_prediction(training_data, test_data, K=K, verbose=args.v, L=args.L, L2=args.L2, save_model=args.save_model,model_path=args.model_path, use_nn=True)
    else:
        assert 'Model not supported'
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
        print_stats(predictions)
        filename = TARGET_FOLDER + '/submission_{}_{}_{}_{}.csv'.format(USERNAME, time.strftime('%c').replace(':','-')[4:-5], args.K, args.L)
        write_data(filename, predictions)
        return predictions


if __name__ == '__main__':
    import sys
    main(sys.argv[1:])
