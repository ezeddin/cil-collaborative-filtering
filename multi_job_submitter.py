import os
import numpy as np

def logspace(start, stop, n):
    return np.logspace(np.log10(start), np.log10(stop), n)

def floatListToStr(l):
	if type(l) == list:
		return '[' + ','.join(['{:.4f}'.format(x) for x in l]) + ']'
	elif type(l) == float:
		return '{.4}'.format(l)
	else:
		return str(l)

# those batches have to be lists of lists.
K_batch = [32,146,195]
L_batch = [0.08,0.09,0.10]
L2_batch = list(np.linspace(0.005,0.1,16))
submission = 'bsub -n {} -B -N -W 04:00 -R "rusage[mem=600]" "python3 batch_train.py --K=\'{}\' --L=\'{}\' --L2=\'{}\'"'

if input('Delete all previous score files? (y/n) ') == 'y':
    os.system('rm data/scores_*')

print('Starting submitting jobs:')
for i,k in enumerate(K_batch):
    submission_formatted = submission.format(len(L_batch)*len(L2_batch), floatListToStr(k), floatListToStr(L_batch), floatListToStr(L2_batch))
    print('Submitting job #{:02}: {}'.format(i, submission_formatted))
    os.system(submission_formatted)
