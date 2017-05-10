from matplotlib import cm
import numpy as np
import glob
import pickle
import scipy.io
from operator import itemgetter
import matplotlib.pyplot as plt
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
files = glob.glob('data/grid_search_*.pkl')[:5:2]

ks = []
ls = []
l2s = []
scores = []
for f in files:
    data = pickle.load(open(f, 'rb'))
    ks += data[0]
    ls += data[1]
    l2s += data[2]
    scores += data[3]
M = list(zip(*(ks,ls,l2s,scores)))

print('Gathered data results from the following parameters:')
print('K  | L      | L2     | Score\n---+--------+--------+---------')
for k,l,l2,s in sorted(list(zip(ks,ls,l2s,scores))):
    print('{:2d} | {:.4f} | {:.4f} | {:.5f}'.format(k,l,l2,s))
    
scipy.io.savemat('data/grid.mat', mdict={'k': ks, 'l': ls, 'l2': l2s, 'scores': scores})

# find out dimensionality

names = ['K', 'L', 'L2']
scales = [sorted(list(set(ks))), sorted(list(set(ls))), sorted(list(set(l2s)))]
dim_size = [len(s) for s in scales]
dim0, dim1, dim2 = np.argsort(dim_size)
full = np.empty(tuple(dim_size))
for entry in M:
    full[scales[0].index(entry[0]), scales[1].index(entry[1]), scales[2].index(entry[2])] = entry[3]

plt.close('all')
# if there are not enough points to plot a reasonable 2D grid,
# better just use the long dimension as x axis and plot several curves in 1D
if dim_size[dim0] <= 4 and dim_size[dim1] <= 6:
    plt.figure()
    for i, x0 in enumerate(scales[dim0]):
        # for the dimension with least values, just do different plot figures
        plt.subplot(100 + 10*dim_size[dim0] + i+1)
        handles = []
        for x1 in scales[dim1]:
            t = list(zip(*sorted([x for x in M if (x[dim0] == x0 and x[dim1] == x1)], key=itemgetter(dim2))))
            handle, = plt.plot(t[dim2], t[3], label='{}={}, {}={}'.format(names[dim0], x0, names[dim1], x1))
            plt.xlabel(names[dim2])
            handles.append(handle)
        plt.legend(handles=handles)
        plt.title('{} = {}'.format(names[dim0], x0))
    plt.show()
# if there ARE enough points to plot a reasonable 2D grid,
else:
    assert False, 'function not ready yet'
    fig = plt.figure(figsize=plt.figaspect(2)*1)
    ax = fig.add_subplot(111, projection='3d')
    for i, x0 in enumerate(scales[dim0]):
        # for the dimension with least values, paint multiple 2D densitiy plots
        t = list(zip(*sorted([x for x in M if (x[dim0] == x0 and x[dim1] == x1)], key=itemgetter(dim2))))
        plot_2D(...)
        ax.contourf(X, Y, A1, 100, zdir='z', offset=0)
    plt.xlabel('K')
    plt.ylabel('L')
    plt.zlabel('L2')
    plt.show()

#%matplotlib qt

# interpolate the data (since it is not equally distributed: used logspace)
#n_interp = 30
#kk = np.linspace(min(k_scale),max(k_scale),n_interp)
#ll = np.linspace(min(l_scale),max(l_scale),n_interp)
#f = scipy.interpolate.interp2d(k_scale, l_scale, z, kind='cubic')

#zz = f(kk, ll)

#plt.imshow(zz.T, vmin=zz.min(), vmax=zz.max(), origin='lower',
#           extent=[min(k_scale), max(k_scale), min(l_scale), max(l_scale)],
#           aspect='auto', cmap=cm.jet)
#plt.xlabel('K')
#plt.ylabel('L')
#plt.colorbar(cmap=cm.jet)
#plt.show()


