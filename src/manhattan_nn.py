import os
import sys
import numpy as np
import pandas as pd
import statistics

def cal_dist(x, y):
    return np.sum(np.abs(x - y))


def find_knn_trial(X_org, X_gen, k=1, n_rand=400):
    ans = []
    X_org = X_org.reshape(X_org.shape[0], -1)
    i = 0
    X_gen = X_gen[np.random.randint(0, X_gen.shape[0], n_rand)]
    print(X_gen.shape)

    # ans = distance.cdist(X_org, X_gen, 'cityblock')
    for x in X_gen:
        dist = np.min(np.sum(np.abs(X_org - x), axis=1))
        ans.append(dist)
        i += 1
        if i % 200 == 0:
            print(i)
    return ans, np.mean(np.sum(X_gen, axis=1)), np.mean(np.sum(X_org, axis=1))


if __name__ == "__main__":
	x_train_headers = list(np.load("../pretrain/x_train.types", allow_pickle = True).keys())
	
	fn_generated = "../synthetic/x_train_binary_synthetic.npy"
	x_train_generated = np.load(fn_generated)

	fn_real = "../pretrain/x_train.matrix"
	x_train_real = np.load(fn_real, allow_pickle = True)

	ans, man_gen, man_org = find_knn_trial(X_org = x_train_real, X_gen = x_train_generated)
	print("Mean Distance generated: ", man_gen)
	print("Mean Distance original:", man_org)
	print("Minimum: ", min(ans))
	print("Maximum: ", max(ans))
	print("Average: ", sum(ans)/len(ans))
	print("Median: ", statistics.median(ans))

