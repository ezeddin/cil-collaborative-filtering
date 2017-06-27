params = '--model=SGD+ --cv_splits=14 --score_averaging=5 --param={} --L={:.4} --external_matrix=True --L2={:.4} --subtract_mean=False'


K_batch = [3, 5,6,8,10,12,15]
L_batch = list(np.linspace(0.03,0.10,8))
L2_batch = list(np.linspace(0,0.16,5))