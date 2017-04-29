import os
import numpy as np

K = list(range(20, 41, 5))
lr_factors = [0.8 , 1, 1.5, 2, 2.5, 3, 3.5]

submission = 'bsub -n 1 -W 00:30 -R "rusage[mem=700]" "python3 train.py --cv_splits=8 --param={} --L=0.084 --lr_factor={} "'

if input('Delete all previous score files? (y/n) ') == 'y':
    os.system('rm data/scores_*')

print('Starting submitting jobs:')
i = 0
for k in K:
    k_list = '[{}]'.format(k)
    for lrf in lr_factors:
	    print('Submitting job #{:02} for K={}, lrf={}: {}'.format(i, k_list, lrf,  submission.format(k_list, lrf)))
	    os.system(submission.format(k_list, lrf))
	    i += 1
