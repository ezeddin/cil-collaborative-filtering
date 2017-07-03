from matplotlib import cm
import numpy as np
import glob
import pickle
import scipy.ndimage
import scipy.io
from operator import itemgetter
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from scipy.interpolate import Rbf
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator
import sys
#%matplotlib inline

def logspace(start, stop, n):
    #return np.logspace(np.log10(start), np.log10(stop), n)
    return start + np.power(np.linspace(0,np.sqrt(stop-start),n), 2)


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

print('\n\nTop 20:')
print('K   | L      | L2     | Score\n---+--------+--------+----------')
for k,l,l2,s in sorted(M, key=itemgetter(3))[:20]:
    print('{:3d} | {:.4f} | {:.4f} | {:.6f}'.format(k,l,l2,s))

sys.exit()

plt.close('all')
slider_k = None
im = None
rbf = None



if dim_size[dim0] == 1:
    # if one dimension is just a scalar, just plot the obvious: a 2D grid
    fig = plt.figure()
    ax = plt.gca()
    # produce interpolated 2d grid
    n = 20
    x_interp = np.linspace(min(scales[dim1]), max(scales[dim1]), (dim_size[dim1]-1)*n+1,)
    y_interp = np.linspace(min(scales[dim2]), max(scales[dim2]), (dim_size[dim2]-1)*n+1,)
    v0,v1 = np.meshgrid(x_interp, y_interp)
    sub_data = list(zip(*M))
    rbf = Rbf(sub_data[dim1], sub_data[dim2], sub_data[3], epsilon=0.05)
    data = rbf(v0, v1)
    levels = logspace(np.min(data), np.max(data), 10)
    #cset = plt.contour(data, levels, linewidths=2, cmap=cm.hot, extent=[min(scales[dim2]), max(scales[dim2]),
    #                                          min(scales[dim1]), max(scales[dim1])])
    #plt.clabel(cset,inline=True,fmt='%.5f',fontsize=8)
    im = ax.imshow(full.T.squeeze(),cmap=cm.RdBu, extent=[min(scales[dim2]), max(scales[dim2]),
                                              min(scales[dim1]), max(scales[dim1])], aspect='auto')
    plt.colorbar(im) # adding the colorbar on the right
    plt.xlabel(names[dim2])
    plt.ylabel(names[dim1])
    plt.title('{} = {}'.format(names[dim0], scales[dim0][0]))
    plt.show()
elif dim_size[dim0] <= 4 and dim_size[dim1] <= 6:
    # if there are not enough points to plot a reasonable 2D grid,
    # better just use the long dimension as x axis and plot several curves in 1D
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
else:
    # if there ARE enough points to plot a reasonable 3D grid
    # and fix the dimension order to K, L, L2
    dim0, dim1, dim2 = [0,1,2]
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    left, bottom, width, height = 0.15, 0.02, 0.7, 0.10
    plt.subplots_adjust(left=left, bottom=0.25)
    
    # show a cut through a selectable K value:
    rbf = Rbf(ks, ls, l2s, scores, epsilon=0.01)
    #im = ax1.imshow(np.zeros((dim_size[dim1], dim_size[dim2])), cmap=cm.RdBu, aspect='auto')
    def update(val):
        x_interp = np.linspace(min(scales[dim1]), max(scales[dim1]), 40)
        y_interp = np.linspace(min(scales[dim2]), max(scales[dim2]), 40)
        data = np.empty((40,40))
        for i in range(40):
            for j in range(40):
                data[i,j] = fn([val, x_interp[i], y_interp[j]])
        #levels = logspace(np.min(data), np.max(data), 10)
        #cset = ax1.contour(data, levels, linewidths=2, cmap=cm.hot, extent=[min(scales[dim2]), max(scales[dim2]),
        #                                          min(scales[dim1]), max(scales[dim1])])
        #ax1.clabel(cset,inline=True,fmt='%.5f',fontsize=8)
        im.set_data(data)
        fig.canvas.draw_idle()
    
    x_interp = np.linspace(min(scales[dim1]), max(scales[dim1]), 40)
    y_interp = np.linspace(min(scales[dim2]), max(scales[dim2]), 40)
    fn = RegularGridInterpolator((scales[dim0],scales[dim1],scales[dim2]), full, method='linear')
    data = np.empty((40,40))
    for i in range(40):
        for j in range(40):
            data[i,j] = fn([25.0, x_interp[i], y_interp[j]])
    #im = ax1.imshow(data, cmap=cm.RdBu, aspect='auto')
    im = ax1.imshow(data,cmap=cm.RdBu, extent=[min(scales[dim2]), max(scales[dim2]),
                                                  min(scales[dim1]), max(scales[dim1])], aspect='auto')
    plt.xlabel(names[dim2])
    plt.ylabel(names[dim1])
    plt.colorbar(im)
    slider_ax = plt.axes([left, bottom, width, height])
    slider_k = Slider(slider_ax, 'K value', min(scales[dim0]), max(scales[dim0]), valinit=min(scales[dim0]))
    slider_k.on_changed(update)
    plt.show()
    
    # produce stacked 2D images in 3D graph
    ax2 = fig.add_subplot(122, projection='3d')
    for i, x0 in enumerate(scales[dim0]):
        # for the dimension with least values, paint multiple 2D densitiy plots
        matrix_dim_reduction = [':']*3
        matrix_dim_reduction[dim0] = str(i)
        t = eval('full[{}]'.format(','.join(matrix_dim_reduction)))
        #t = scipy.ndimage.zoom(t, 3)
        x,y = np.meshgrid(scales[dim1], scales[dim2])
        levels = logspace(np.min(t), np.max(t), 20)
        norm = norm = mc.BoundaryNorm(levels, 256)
        cset = ax2.contourf(x, y, t.T, zdir='z', offset=x0,
                            alpha=0.5, norm=norm, cmap=cm.RdBu,
                            levels=levels, vmin=np.min(full), vmax=np.max(full))
        ax2.set_zlim((min(scales[0]), max(scales[0])))
    plt.colorbar(cset)
    ax2.set_zlabel(names[dim0])
    ax2.set_xlabel(names[dim1])
    ax2.set_ylabel(names[dim2])
    plt.show()
