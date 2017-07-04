import glob
import pickle
import scipy.io
from operator import itemgetter
import matplotlib.pyplot as plt
#%matplotlib inline

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

k_scale = sorted(list(set(ks)))
l_scale = sorted(list(set(ls)))
l2_scale = sorted(list(set(l2s)))
M = list(zip(*(ks,ls,l2s,scores)))
scipy.io.savemat('data/grid.mat', mdict={'k': ks, 'l': ls, 'l2': l2s, 'scores': scores})

print('Gathered data results from the following parameters:')
print('K  | L      | L2     | Score\n---+--------+--------+---------')
for k,l,l2,s in sorted(list(zip(ks,ls,l2s,scores))):
    print('{:2d} | {:.4f} | {:.4f} | {:.5f}'.format(k,l,l2,s))


#%matplotlib qt

plt.figure(1)
plt.subplot(211)

# print score in function of L
handles = []
for k in [26,32,146,195]:
    for i, other in enumerate([0.1, 0.5]):
        t = list(zip(*sorted([x for x in M if (x[0] == k and x[1] == other)], key=itemgetter(2))))
        handle, = plt.plot(t[2], t[3], label='K={} L={}'.format(k, other))
        plt.xlabel('L2')
        handles.append(handle)
plt.legend(handles=handles)

# print score in function of L2
plt.subplot(212)
handles = []
for k in [26,32,146,195]:
    for i, other in enumerate([0.1, 0.5]):
        t = list(zip(*sorted([x for x in M if (x[0] == k and x[2] == other)], key=itemgetter(1))))
        handle, = plt.plot(t[1], t[3], label='K={} L2={}'.format(k, other))
        handles.append(handle)
        plt.xlabel('L')
plt.legend(handles=handles)

plt.show()



scipy.io.savemat('data/grid.mat', mdict={'k': list(M[0]), 'l': list(M[1]),
                                         'l2': list(M[2]), 'scores': list(M[3])})
