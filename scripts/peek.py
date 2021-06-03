import joblib
import os
import sys


for filename in sys.argv[1:]:
    with open(filename, 'rb') as f:
        data = joblib.load(f)

    clf = data['est']
    print(f'{filename}:', clf.coef_.shape)
