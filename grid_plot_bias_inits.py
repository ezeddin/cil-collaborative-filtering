from matplotlib import cm
import numpy as np
import glob
import pickle
import scipy.io
from operator import itemgetter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#%matplotlib inline

def print_2D(orig_scale0, orig_scale1, interp_factors, full_matrix):
    # interpolate the data (since it is not equally distributed: used logspace)
    # https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html
    n0 = len(orig_scale0)*interp_factors[0]
    n1 = len(orig_scale1)*interp_factors[1]
    kk = np.linspace(min(orig_scale0),max(orig_scale0),n0)
    ll = np.linspace(min(orig_scale1),max(orig_scale1),n1)
    f = scipy.interpolate.interp2d(orig_scale0, orig_scale1, full_matrix, kind='cubic')
    zz = f(kk, ll)
    #plt.imshow(zz.T, vmin=zz.min(), vmax=zz.max(), origin='lower',
    #           extent=[min(orig_scale0), max(orig_scale0), min(orig_scale1), max(orig_scale1)],
    #           aspect='auto', cmap=cm.jet)
    ax.contourf(X, Y, A1, 100, zdir='z', offset=0)
    # TODO: craft the colormap like in the matlab script
    plt.colorbar(cmap=cm.jet)

# import data
files = glob.glob('data/grid_search_*.pkl')

ks = []
ls = []
l2s = []
bss = []
scores = []
for f in files:
    data = pickle.load(open(f, 'rb'))
    ks += data[0]
    ls += data[1]
    l2s += data[2]
    bss += data[3]
    scores += data[4]
M = list(zip(*(ks,ls,l2s,bss,scores)))

print('Gathered data results from the following parameters:')
print('K  | L      | L2     | Score\n---+--------+--------+---------')
#for k,l,l2,s in sorted(list(zip(ks,ls,l2s,scores))):
#    print('{:2d} | {:.4f} | {:.4f} | {:.5f}'.format(k,l,l2,s))
    
# find out dimensionality

names = ['K', 'L', 'L2', 'BS']
scales = [sorted(list(set(ks))), sorted(list(set(ls))), sorted(list(set(l2s))), sorted(list(set(bss)))]
dim_size = [len(s) for s in scales]
full = np.ones((dim_size[2],dim_size[3]))*np.nan
for entry in M:
    full[scales[2].index(entry[2]), scales[3].index(entry[3])] = entry[4]# if entry[4] < 1.3 else np.NaN

scipy.io.savemat('data/grid.mat', mdict={'UVs': ks, 'UVm': ls, 'L2': l2s, 'Bs': bss, 'scores': scores,
                                         'full':full, 'l2_set': scales[2], 'bs_set':scales[3]})