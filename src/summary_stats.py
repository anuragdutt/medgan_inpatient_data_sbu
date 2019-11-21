import os
import sys
import numpy as np
import pandas as pd
import _pickle as pickle


def convertTo3DigitIcd9(dxStr):
	return "D_" + str(dxStr)[:3]	


def transformHeaders(diag_icd_file, headers):
	diag = pd.read_csv(diag_icd_file)
	diag['icd_3digits'] = diag['icd_codes'].apply(convertTo3DigitIcd9)

	diag = diag.sort_values(['pid', 'visit_id'], ascending = True)
	diag['pid_vid'] = diag['pid'].apply(lambda x: str(x)) + "-" + diag['visit_id'].apply(lambda x: str(x))

	groups = diag.groupby('pid')

	binary_array = np.array([])
	count_array = np.array([])
	cnt = 0
	for key in groups.groups.keys():
		icd = diag['icd_3digits'].ix[groups.groups[key]].tolist()
		bin_arr = np.isin(np.array(ch), np.array(icd)).astype(int)
		indexes = np.where(bin_arr)
		count_arr = np.array(bin_arr.tolist())
		count = [icd.count(e) for e in np.array(ch)[indexes]]
		count_arr[indexes] = count
		cnt = cnt + 1
		print(cnt)
		if cnt == 1:
			binary_array = np.append(binary_array, bin_arr)
			count_array = np.append(count_array, count_arr)
		else:
			binary_array = np.vstack([binary_array, bin_arr])
			count_array = np.vstack([count_array, count_arr])


	return binary_array, count_array

if __name__ == "__main__":

	count_headers = np.load("../pretrain/count.types", allow_pickle = True)
	binary_headers = np.load("../pretrain/binary.types", allow_pickle = True)

	ch = list(count_headers.keys())
	bh = list(binary_headers.keys())

	# diag_icd_file = "../generated/DIAGNOSES_ICD_GENERATED_TRAIN.csv"


	# ba, ca = transformHeaders(diag_icd_file, ch)

	# pickle.dump(ba, open("../generated/train_binary.matrix", 'wb'), -1)
	# pickle.dump(ca, open("../generated/train_count.matrix", 'wb'), -1)
	


	ba = np.load("../generated/train_binary.matrix", allow_pickle = True)
	ca = np.load("../generated/train_count.matrix", allow_pickle = True)
	print(ba.shape)
	print(ca.shape)


	binary_file = "../synthetic/cerner_binary.npy" 
	binary_dat = np.load(binary_file)
	count_file = "../synthetic/cerner_count.npy" 
	count_dat = np.load(count_file)

	print(binary_dat.shape)
	print(count_dat.shape)

	ctrain = pd.DataFrame(data = ca,
						columns = ch)
	btrain = pd.DataFrame(data = ba,
						columns = bh)

	cgenerated = pd.DataFrame(data = count_dat,
						columns = ch)
	bgenerated = pd.DataFrame(data = binary_dat,
						columns = bh)
	bgenerated[bgenerated >= 0.5] = 1
	bgenerated[bgenerated < 0.5] = 0

	count_train_means = ctrain.mean(axis = 0)
	count_generated_means = cgenerated.mean(axis = 0)
	count_error = abs(count_generated_means - count_train_means)
	mae_count = count_error.mean()

	binary_train_means = btrain.mean(axis = 0)
	binary_generated_means = bgenerated.mean(axis = 0)
	binary_error = abs(binary_generated_means - binary_train_means)
	mae_binary = binary_error.mean()

	print(count_generated_means[count_generated_means == 0].shape)
	print(binary_generated_means[binary_generated_means == 0].shape)
	print(mae_binary, mae_count)


	# tmp = binary_dat[1]
	# tmp = tmp[tmp >= 0.01]
	# print(tmp)
	# # print(binary_dat[0])
	# exit(0)
	# count_file = "../synthetic/cerner_count.npy" 
	# count_dat = np.load(count_file)
	# print(count_dat.shape)

	# print(count_dat)
