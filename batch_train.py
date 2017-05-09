import train
import multiprocessing
import datetime
import os
import numpy as np
import scipy.io
import pickle
import time
import sys
import argparse


"""
from http://stackoverflow.com/questions/15414027/multiprocessing-pool-makes-numpy-matrix-multiplication-slower
"""

params = '--model=SGD+ --cv_splits=14 --score_averaging=5 --external_matrix=True --param={} --L={:.4} --L2={:.4} --subtract_mean=False'

raw_data = None

def worker_function(result_queue, worker_index, k, l, l2):
    # Work on a certain task of the grid search
    print('Executing train.py ' + params.format(k,l,l2))
    result_queue.put((worker_index, k, l, l2, train.main(params.format(k,l,l2).split(), raw_data)))


def work(K, L, L2, dry_run):
    # after importing numpy, reset the CPU affinity of the parent process so
    # that it will use all cores
    #os.system("taskset -p 0xff %d" % os.getpid())
    result_queue = multiprocessing.Queue()

    # Prepare child processes.
    children = []
    i = 0
    # start processes for every L value in two different K values
    for k in K:
        for l in L:
            for l2 in L2:
                print('Starting process: {}'.format(params.format(k,l,l2)))
                if not dry_run:
                    children.append(multiprocessing.Process(target=worker_function,args=(result_queue, i, k, l, l2)))
                i += 1

    if dry_run:
        return

    # Run child processes.
    print('Starting {} processes...'.format(len(K)*len(L)*len(L2)))
    for c in children:
        # also kill child processes if parent is killed
        c.daemon=True
        c.start()
    # Wait for all results to arrive and gather in a sparse manner: (k,l,score(k,l))
    K = []
    L = []
    L2 = []
    Scores = []
    print('Waiting for processes to return...')
    for _ in range(i):
        ind, k, l, l2, result = result_queue.get(block=True)
        print('Process #{:03} with datapoint ({:2}, {:.4}) returned {}'.format(ind, k, l, result[0][0]))
        K.append(k)
        L.append(l)
        L2.append(l2)
        Scores.append(result[0][1])
    # Join child processes (clean up zombies).
    for c in children:
        c.join()
    return (K, L, L2, Scores)


def main():
    global raw_data

    parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--K', type=str, default='12',
                        help='Provide list of K parameters, write as python expression.')
    parser.add_argument('--L', type=str, default='[0.05,0.1]',
                        help='Provide list of L parameters, write as python expression.')
    parser.add_argument('--L2', type=str, default='[0.02,0.04]',
                        help='Provide list of L2 parameters, write as python expression.')
    parser.add_argument('--dry_run', type=bool, default=False,
                        help='Only print the commands executed.')
    args = parser.parse_args()

    print('Loading dataset once:')
    raw_data = train.load_data(train.DATA_FILE)

    try:
        args.K = eval(args.K) if type(eval(args.K)) == list else [eval(args.K)]
        args.L = eval(args.L) if type(eval(args.L)) == list else [eval(args.L2)]
        args.L2 = eval(args.L2) if type(eval(args.L2)) == list else [eval(args.L2)]
    except Exception as e:
        print('Couldn\'t convert arguments to python expression')
        raise e

    if len(args.K)*len(args.L)*len(args.L2) > 48:
        print('Warning: requires more than 48 processes!')

    t0 = datetime.datetime.now()
    print('Starting workers for K={}, L={}, L2={}'.format(args.K, args.L, args.L2))
    result = work(args.K, args.L, args.L2, args.dry_run)
    duration = datetime.datetime.now() - t0
    print('Finished. Duration: {}'.format(str(duration).split('.')[0]))
    filename = 'data/grid_search_{}'.format(time.strftime('%c').replace(':','-')[4:-5])
    
    print('Saving result in pickle file...')
    pickle.dump(result, open(filename+'.pkl', 'wb'))
    print('Saving result in matlab file...')
    scipy.io.savemat(filename+'.mat', mdict={'k': result[0], 'l': result[1], 'l2': result[2], 'scores': result[3]})


if __name__ == '__main__':
    main()
