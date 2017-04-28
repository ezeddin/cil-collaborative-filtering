import os
import numpy as np
from time import sleep

K = list(range(3,20))
L = np.logspace(-3, -0.3, 15)

submission = 'bsub -n 1 -W 00:30 -R "rusage[mem=512]" "python3 train.py --model=SGD --cv_splits=14 --param={} --L={:.3}"'

i = 0
for k in K:
    for l in L:
        print('submitting for K={} and L={:.3}: {}\n'.format(k, l, submission.format(k, l)))
        os.system(submission.format(k, l))
        sleep(0.3)
