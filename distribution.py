import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# code from http://danielhnyk.cz/fitting-distribution-histogram-using-python/

DATA_FILE = input('please enter a valid file path: ')

print("Loading values from training set...")
with open(DATA_FILE, 'r') as f:
    lines = f.readlines()
    nb_lines = len(lines) - 1
    x = np.empty(nb_lines, dtype='float')
    for i, l in enumerate(lines[1:]):
        x[i] = float(l.split(',')[1].strip())

print('saving to file..')
scipy.io.savemat('data/values.mat', mdict={'x': x})

# plot normed histogram
plt.hist(x, 50, normed=True)
