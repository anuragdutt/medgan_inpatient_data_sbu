import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import 


def RandomForestClassification(train_mat, test_mat, headers):
	train = pd.DataFrame(data = train_mat, columns = headers)
	test = pd.DataFrame(data = test_mat, columns = headers)

	for col in headers:
		x_train = train.drop([col], axis = 1)
		y_train = train.loc[:, col]
		rf = RandomForestClassifier(n_estimators = 10000, random_state = 0, n_jobs = -1)
		rf.fit(x_train, y_train)

		

if __name__ == "__main__":
	count_headers = np.load("../pretrain/count.types", allow_pickle = True)
	binary_headers = np.load("../pretrain/binary.types", allow_pickle = True)

	ch = list(count_headers.keys())
	bh = list(binary_headers.keys())



	binary_file_generated = "../synthetic/cerner_binary.npy" 
	binary_generated = np.load(binary_file_generated)
	count_file_generated = "../synthetic/cerner_count.npy" 
	count_generated = np.load(count_file_generated)

############Test

	ba_test = np.load("../generated/test_binary.matrix", allow_pickle = True)
	ca_test = np.load("../generated/test_count.matrix", allow_pickle = True)


