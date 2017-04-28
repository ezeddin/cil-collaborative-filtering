import os
import numpy as np

K = list(range(2,100,2))
L = np.logspace(np.log10(0.0005), np.log10(0.2), 20)
submission = 'bsub -n 1 -W 00:30 -R "rusage[mem=512]" "python3 train.py --model=SGD --cv_splits=14 --score_averaging=2 --param={} --L={:.5}"'

if input('Delete all previous score files? (y/n) ') == 'y':
    os.system('rm data/scores_*')

print('Starting submitting jobs:')
i = 0
for k in K:
    for l in L:
        print('Submitting for K={} and L={:.3}: {}'.format(k, l, submission.format(k, l)))
        os.system(submission.format(k, l))