import os
import numpy as np

K = list(range(2,194,4))
submission = 'bsub -n {} -W 04:00 -R "rusage[mem=600]" "python3 batch_train.py {}"'

L_per_K = 15

if input('Delete all previous score files? (y/n) ') == 'y':
    os.system('rm data/scores_*')

print('Starting submitting jobs:')
i = 0
for k in K:
    k_list = '[{},{}]'.format(k,k+2)
    print('Submitting job #{:02} for K={}: {}'.format(i, k_list, submission.format(k_list)))
    os.system(submission.format(len(K)*L_per_K, k_list))
    i += 1
