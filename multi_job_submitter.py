# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 17:41:58 2017

@author: cyril
"""

import os
import numpy as np

K = list(range(3,20))
L = np.logspace(-3, -0.3, 15)

submission = 'bsub -n 1 -W 00:30 -R "rusage[mem=512]" "python3 train.py --model=SGD --cv_splits=14 --param={} --L={:.3}"'

for k in K:
    for l in L:
        print(submission.format(k, l))
        #os.system(submission.format(k, l))
