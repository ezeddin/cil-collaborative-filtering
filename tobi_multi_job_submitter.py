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
#K_batch = [3, 5,8,10,12,15,20]
#L_batch = list(np.linspace(0.01,0.13,7))
#L2_batch = list(np.linspace(0.01,0.13,7))
K_batch = [8]
L_batch = [0.05,0.08]
L2_batch = [0.03,0.04,0.06]

os.system('module load python/3.3.3')

submission = 'bsub -n {} -B -N -W 04:00 -R "rusage[mem=1000]" "python3 batch_train.py --K=\'{}\' --L=\'{}\' --L2=\'{}\' "'

if input('Delete all previous score files? (y/n) ') == 'y':
    os.system('rm data/scores_*')

print('Starting submitting jobs:')
i = 0
for k in K_batch:
    submission_formatted = submission.format(len(L_batch)*len(L2_batch), k, floatListToStr(L_batch), floatListToStr(L2_batch))
    print('Submitting job #{:02}: {}'.format(i, submission_formatted))
    os.system(submission_formatted)
    i += 1