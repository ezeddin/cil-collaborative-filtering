# -*- coding: utf-8 -*-

DATA_FILE = 'data/data_train.csv'
SUBMISSION_EXAMPLE = 'data/sampleSubmission.csv'

print("Loading indices set from given data")
given_set = set()
with open(DATA_FILE, 'r') as f:
    lines = f.readlines()
        
    for i, l in enumerate(lines[1:]):
        s = l.split(',')
        c = s[0].split('_')
        row = int(c[1][1:]) - 1 # 0-based indexing
        col = int(c[0][1:]) - 1 # 0-based indexing
        given_set.add((row,col))

print("Loading indices set from asked data")
asked_set = set()
with open(SUBMISSION_EXAMPLE, 'r') as example_file:
    lines = example_file.readlines()
    
    for i, l in enumerate(lines[1:]):
        s = l.split(',')
        c = s[0].split('_')
        item = int(c[1][1:]) - 1 # 0-based indexing
        user = int(c[0][1:]) - 1 # 0-based indexing
        asked_set.add((item,user))

print('given set includes {} elements'.format(len(given_set)))
print('asked set includes {} elements'.format(len(given_set)))
print('the intersection of the two sets contain {} elements'.format(len(given_set & asked_set)))
