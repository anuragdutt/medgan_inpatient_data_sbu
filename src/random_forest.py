import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# from sklearn.metrics import 


def randomForestClassification(train_mat, test_mat, headers, binary = False):
	train = pd.DataFrame(data = train_mat, columns = headers)
	test = pd.DataFrame(data = test_mat, columns = headers)

	if binary:
		train[train >= 0.5] = 1
		train[train < 0.5] = 0

		test[test >= 0.5] = 1
		test[test < 0.5] = 0

	ret_list = []
	count = 0

	for col in headers:
		count = count + 1
		print(count)

		x_train = train.drop([col], axis = 1)
		y_train = train.loc[:, col]

		x_test = test.drop([col], axis = 1)
		y_test = test.loc[:, col]

		rf = RandomForestClassifier(n_estimators = 300, random_state = 0, n_jobs = -1)
		# print("Starting training")
		trd = rf.fit(x_train, y_train)
		# print("Ending Training")
		y_pred = rf.predict(x_test)

		f1 = f1_score(y_test, y_pred)
		acc = accuracy_score(y_test, y_pred)

		prob_true = sum(y_test)/len(y_test)
		prob_pred = sum(y_pred)/len(y_pred)

		retl = [col, acc, f1, prob_true, prob_pred]
		ret_list.append(retl)
		# x_test = train.drop([col], axis = 1)
		# y_train = train.loc[:, col]
		# rf = RandomForestClassifier(n_estimators = 10000, random_state = 0, n_jobs = -1)
		# if count == 5:
		# 	break


	retdf = pd.DataFrame(ret_list, columns = ['variable', "f1", "accuracy", "prob_occurence_true", "prob_occurence_pred"])
	return retdf



if __name__ == "__main__":
	# count_headers = np.load("../pretrain/count.types", allow_pickle = True)
	headers_dict = np.load("../pretrain/x_train_filtered_01.types", allow_pickle = True)
	bh = list(headers_dict.keys())

	filename_generated = "../synthetic/x_train_filtered_01_synthetic.npy" 
	file_generated = np.load(filename_generated)
	
	filename_test = "../pretrain/x_test_filtered_01.matrix"
	file_test = np.load(filename_test, allow_pickle = True)

	df = randomForestClassification(train_mat = file_generated, test_mat = file_test, headers = bh, binary = True)
	df.to_csv("../summary_stats/random_forest_metrics.csv", index = False)
	# print(file_test.shape)


############Test



