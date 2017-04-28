import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import cm
import scipy.interpolate

files = glob.glob('data/scores_*')

ls = []
ks = []
scores = []
for f in files:
    ls.append(float(f.split('_')[-1][:-4]))
    data = pickle.load(open(f, 'rb'))
    scores.append(data[0,1])
    ks.append(data[0,0])

k_scale = sorted(list(set(ks)))
l_scale = sorted(list(set(ls)))

z = np.empty((len(k_scale), len(l_scale)))
for l,k,s in zip(ls, ks, scores):
    z[k_scale.index(k), l_scale.index(l)] = s

plt.imshow(z.T, vmin=z.min(), vmax=z.max(), origin='lower',
           extent=[min(k_scale), max(k_scale), min(l_scale), max(l_scale)],
           aspect='auto', cmap=cm.jet)
plt.xlabel('K')
plt.ylabel('L')
plt.colorbar(cmap=cm.jet)
plt.show()

scipy.io.savemat('data/grid.mat', mdict={'k': k_scale, 'l': l_scale, 'scores': z})
