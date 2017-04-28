import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import scipy.io

# code from http://danielhnyk.cz/fitting-distribution-histogram-using-python/

DATA_FILE = input('please enter a valid file path: ')

print("Loading values from training set")
with open(DATA_FILE, 'r') as f:
    lines = f.readlines()
    nb_lines = len(lines) - 1
    x = np.empty(nb_lines, dtype='float')
    for i, l in enumerate(lines[1:]):
        x[i] = float(l.split(',')[1].strip())

scipy.io.savemat('data/values.mat', mdict={'x': x})

# plot normed histogram
plt.hist(x, normed=True)

lnspc = np.linspace(1.0, 5.0, len(x))

# lets try the normal distribution first
m, s = stats.norm.fit(x) # get mean and standard deviation
pdf_g = stats.norm.pdf(lnspc, m, s) # now get theoretical values in our interval
plt.plot(lnspc, pdf_g, label="Norm") # plot it

# exactly same as above
ag,bg,cg = stats.gamma.fit(x)
pdf_gamma = stats.gamma.pdf(lnspc, ag, bg,cg)
plt.plot(lnspc, pdf_gamma, label="Gamma")

# guess what :)
ab,bb,cb,db = stats.beta.fit(x)
pdf_beta = stats.beta.pdf(lnspc, ab, bb,cb, db)
plt.plot(lnspc, pdf_beta, label="Beta")
