import train
import multiprocessing
import datetime
import os
import numpy as np
import scipy.io
import pickle
import time
import sys

"""
from http://stackoverflow.com/questions/15414027/multiprocessing-pool-makes-numpy-matrix-multiplication-slower
"""

params = '--model=SGD --cv_splits=14 --score_averaging=3 --external_matrix=True --param={} --L={:.5}'

raw_data = None

def worker_function(result_queue, worker_index, k, l):
    # Work on a certain task of the grid search
    result_queue.put((worker_index, k, l, train.main(params.format(k,l).split(), raw_data)))


def work(K_chunk):
    # after importing numpy, reset the CPU affinity of the parent process so
    # that it will use all cores
    #os.system("taskset -p 0xff %d" % os.getpid())
    result_queue = multiprocessing.Queue()
    
    # Prepare child processes.
    children = []
    i = 0
    # start processes for every L value in two different K values
    for k in K_chunk:
        if k < 30:
            L=np.linspace(0.03,0.11,15)+0.006/30*k
        else:
            L=np.linspace(0.03,0.11,15)+0.006 
        for l in L:
            children.append(
                multiprocessing.Process(
                    target=worker_function,
                    args=(result_queue, i, k, l)
                    )
                )
            i += 1

    # Run child processes.
    print('Starting processes...')
    for c in children:
        # also kill child processes if parent is killed
        c.daemon=True
        c.start()
    # Wait for all results to arrive and gather in a sparse manner: (k,l,score(k,l))
    K = []
    L = []
    Scores = []
    print('Waiting for processes to return...')
    for _ in range(i):
        ind, k, l, result = result_queue.get(block=True)
        print('Process #{:03} with datapoint ({:2}, {:.4}) returned {}'.format(ind, k, l, result[0][0]))
        K.append(k)
        L.append(l)
        Scores.append(result[0][1])
    # Join child processes (clean up zombies).
    for c in children:
        c.join()
    return (K, L, Scores)


def main(argv):
    global raw_data
    print('Loading dataset once:')
    raw_data = train.load_data(train.DATA_FILE)

    t0 = datetime.datetime.now()
    Ks = eval(argv[0])
    assert type(Ks) == list, 'Please provide a python list as argument!'
    print('Starting workers for K={}'.format(Ks))
    result = work(Ks)
    duration = datetime.datetime.now() - t0
    print('Finished. Duration: {}'.format(str(duration).split('.')[0]))
    filename = 'data/grid_search_{}'.format(time.strftime('%c').replace(':','-')[4:-5])
    
    print('Saving result in pickle file...')
    pickle.dump(result, open(filename+'.pkl', 'wb'))
    print('Saving result in matlab file...')
    scipy.io.savemat(filename+'.mat', mdict={'k': result[0], 'l': result[1], 'scores': result[2]})


if __name__ == '__main__':
    main(sys.argv[1:])
