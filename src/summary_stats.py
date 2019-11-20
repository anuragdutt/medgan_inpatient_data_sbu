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

	diag_icd_file = "../generated/DIAGNOSES_ICD_GENERATED_TRAIN.csv"


	ba, ca = transformHeaders(diag_icd_file, ch)

	pickle.dump(ba, open("../generated/train_binary.matrix", 'wb'), -1)
	pickle.dump(ca, open("../generated/train_count.matrix", 'wb'), -1)
	
	# t1 = np.load("../generated/train_binary.matrix", allow_pickle = True)
	# t2 = np.load("../generated/train_count.matrix", allow_pickle = True)
	# print(t1.shape)
	# print(t2.shape)
	# print(t1)
	# exit(0)

	# binary_file = "../synthetic/cerner_binary.npy" 
	# binary_dat = np.load(binary_file)
	# print(binary_dat.shape)

	# tmp = binary_dat[1]
	# tmp = tmp[tmp >= 0.01]
	# print(tmp)
	# # print(binary_dat[0])
	# exit(0)
	# count_file = "../synthetic/cerner_count.npy" 
	# count_dat = np.load(count_file)
	# print(count_dat.shape)

	# print(count_dat)
