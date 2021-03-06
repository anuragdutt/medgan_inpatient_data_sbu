import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calProb(dat, headers):
	rows = dat.shape[0]
	dat = pd.DataFrame(dat, columns = headers)
	colProb = dat.sum(axis = 0, skipna = True)/rows
	return colProb


def plotProb(prob_real, prob_generated, save_path):

	df_real = pd.DataFrame({'icd':prob_real.index, 'probabilities_real':prob_real.values})
	df_generated = pd.DataFrame({'icd':prob_generated.index, 'probabilities_generated':prob_generated.values})
	
	df = pd.merge(df_real, df_generated, on = "icd")
	df = df.sort_values(by=['probabilities_real'], ascending = True)
	# prob = prob.sort_values(ascending = True)


	graph = plt.figure()
	plt.scatter(x = df['probabilities_real'], y = df['probabilities_generated'], s = 3, color = "red", alpha=0.8)
	# plt.plot([0, max(df['probabilities_real'])], [0, max(df['probabilities_generated'])], color = 'blue')
	plt.plot([0, 1], [0, 1], color = 'blue')
	plt.title('Dimension-wise probabilities Comparison')
	plt.xlabel('Real Probabilites')
	plt.ylabel('Generated Probabilities')
	graph.savefig(save_path, bbox_inches='tight')


if __name__ == "__main__":
	
	f_generated = sys.argv[1] 
	f_real = sys.argv[2]
	f_headers = sys.argv[3]
	save_path = sys.argv[4]

	# x_train_headers = list(np.load("../pretrain/x_train.types", allow_pickle = True).keys())
	x_train_headers = list(np.load(f_headers, allow_pickle = True).keys())

	# fn_generated = "../synthetic/x_train_binary_synthetic.npy"
	fn_generated = f_generated
	x_train_generated = np.load(fn_generated)
	x_train_generated[x_train_generated >= 0.5] = 1
	x_train_generated[x_train_generated < 0.5] = 0
	prob_generated = calProb(dat = x_train_generated, headers = x_train_headers)

	# fn_real = "../pretrain/x_train.matrix"
	fn_real = f_real
	x_train_real = np.load(fn_real, allow_pickle = True)
	prob_real = calProb(dat = x_train_real, headers = x_train_headers)


	plotProb(prob_real, prob_generated, save_path)