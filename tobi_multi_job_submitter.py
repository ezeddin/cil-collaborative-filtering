import os
import numpy as np




K = [12]
Ls=   [0.04, 0.06,0.08,0.1, 0.12, 0.15 ]
L2s = [0.04, 0.06,0.08,0.1, 0.12, 0.15, 0.18, 0.20]


MODEL = "SGD+"
CV_SPLITS = 14
AVERAGE_OVER = 3

submission = 'bsub -n 1 -W 2:00 -R "rusage[mem=700]" "python3 train.py --model={} --cv_splits={} --score_averaging={} --param={} --L={} --L2={}  "'

if input('Delete all previous score files? (y/n) ') == 'y':
    os.system('rm data/scores_*')

print('Starting submitting jobs:')
i = 0
for k in K:
    for l in Ls:
	    for l2 in L2s:
	    	command = submission.format(MODEL, CV_SPLITS, AVERAGE_OVER, k, l, l2)
	    	print('Submitting job #{:02} for K={}, L={},  L2={}: {}'.format(i, k, l, l2, command))
	    	os.system(command)
	    	i += 1
