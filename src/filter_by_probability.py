import os
import sys
import numpy as np
import pandas as pd


def filterData(dat, headers, prob = 0.01):

	rows = dat.shape[0]
	print(dat.shape	)
	df = pd.DataFrame(dat, columns = headers)
	col_prob = (df.sum(axis = 0, skipna = True) +0.0) /rows
	prob_cutoff = col_prob.index[col_prob.values > prob]
	print(prob_cutoff.shape)


if __name__ == "__main__":
	x_train_headers = list(np.load("../pretrain/x_train.types", allow_pickle = True).keys())

	fn_real = "../pretrain/x_train.matrix"
	x_train_real = np.load(fn_real, allow_pickle = True)


	filterData(x_train_real, x_train_headers, prob = 0.01)