from matplotlib import cm
import numpy as np
import glob
import pickle
import scipy.ndimage
import scipy.io
from operator import itemgetter
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from mpl_toolkits.mplot3d import Axes3D
#%matplotlib inline

def logspace(start, stop, n):
    return np.logspace(np.log10(start), np.log10(stop), n)

# import data
files = glob.glob('data/grid_search_*.pkl')

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
full = np.ones(tuple(dim_size))*np.nan
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
    dim0 = 0
    dim1, dim2 = np.argsort(dim_size[1:]) + 1
    fig = plt.figure()
    #ax1 = fig.add_subplot(121)
    #ax1.imshow(data, cmap=plt.cm.BrBG, interpolation='nearest', origin='lower', extent=[0,1,0,1])
    #ax = fig.add_subplot(111, projection='3d')
    ax2 = fig.add_subplot(111, projection='3d')
    for i, x0 in enumerate(scales[dim0]):
        # for the dimension with least values, paint multiple 2D densitiy plots
        matrix_dim_reduction = [':']*3
        matrix_dim_reduction[dim0] = str(i)
        t = eval('full[{}]'.format(','.join(matrix_dim_reduction)))
        #t = scipy.ndimage.zoom(t, 3)
        x,y = np.meshgrid(scales[dim1], scales[dim2])
        levels = logspace(np.min(full), np.max(full), 20)
        norm = norm = mc.BoundaryNorm(levels, 256)
        cset = ax2.contourf(x, y, t.T, zdir='z', offset=x0,
                            alpha=0.5, cmap = plt.cm.jet, norm=norm,
                            levels=levels, vmin=np.min(full), vmax=np.max(full))
        ax2.set_zlim((min(scales[0]), max(scales[0])))
    plt.colorbar(cset)
    ax2.set_zlabel(names[dim0])
    ax2.set_xlabel(names[dim1])
    ax2.set_ylabel(names[dim2])
    plt.show()
