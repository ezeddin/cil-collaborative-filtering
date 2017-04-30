import os
import numpy as np

L=0.084
K = [3, 6, 10, 15, 20]
L2s = [0.05,0.08,0.1, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]


submission = 'bsub -n 1 -W 2:00 -R "rusage[mem=700]" "python3 train.py --param={} --L=0.084 --L2={} "'

if input('Delete all previous score files? (y/n) ') == 'y':
    os.system('rm data/scores_*')

print('Starting submitting jobs:')
i = 0
for k in K:
    k_list = '[{}]'.format(k)
    for l2 in L2s:
	    print('Submitting job #{:02} for K={}, L2={}: {}'.format(i, k_list, l2,  submission.format(k_list, lrf)))
	    os.system(submission.format(k_list, lrf))
	    i += 1
