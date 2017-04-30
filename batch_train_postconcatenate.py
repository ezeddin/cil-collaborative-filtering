import glob
import pickle
import matplotlib.pyplot as plt
import scipy.io

files = glob.glob('data/grid_search_*.pkl')

ls = []
ks = []
scores = []
for f in files:
    data = pickle.load(open(f, 'rb'))
    ks += data[0]
    ls += data[1]
    scores += data[2]

k_scale = sorted(list(set(ks)))
l_scale = sorted(list(set(ls)))
scipy.io.savemat('data/grid.mat', mdict={'k': ks, 'l': ls, 'scores': scores})

print('Gathered data results from the following parameters:')
print('K  | L      | Score\n---+--------+---------')
for k,l,s in sorted(list(zip(ks,ls,scores))):
    print('{:2d} | {:.4f} | {:.5f}'.format(k,l,s))