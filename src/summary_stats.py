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

	# diag_icd_file_train = "../generated/DIAGNOSES_ICD_GENERATED_TRAIN.csv"
	# ba_train, ca_train = transformHeaders(diag_icd_file_train, ch)
	# pickle.dump(ba_train, open("../generated/train_binary.matrix", 'wb'), -1)
	# pickle.dump(ca_train, open("../generated/train_count.matrix", 'wb'), -1)
	

	# diag_icd_file_test = "../generated/DIAGNOSES_ICD_GENERATED_TEST.csv"
	# ba_test, ca_test = transformHeaders(diag_icd_file_test, ch)
	# pickle.dump(ba_test, open("../generated/test_binary.matrix", 'wb'), -1)
	# pickle.dump(ca_test, open("../generated/test_count.matrix", 'wb'), -1)


	binary_file = "../synthetic/cerner_binary.npy" 
	binary_dat = np.load(binary_file)
	count_file = "../synthetic/cerner_count.npy" 
	count_dat = np.load(count_file)
	print(binary_dat.shape)
	print(count_dat.shape)
	cgenerated = pd.DataFrame(data = count_dat,
						columns = ch)
	bgenerated = pd.DataFrame(data = binary_dat,
						columns = bh)
	bgenerated[bgenerated >= 0.5] = 1
	bgenerated[bgenerated < 0.5] = 0


########Train


	ba_train = np.load("../generated/train_binary.matrix", allow_pickle = True)
	ca_train = np.load("../generated/train_count.matrix", allow_pickle = True)
	print(ba_train.shape)
	print(ca_train.shape)


	ctrain = pd.DataFrame(data = ca_train,
						columns = ch)
	btrain = pd.DataFrame(data = ba_train,
						columns = bh)


	count_train_means = ctrain.mean(axis = 0)
	count_generated_means = cgenerated.mean(axis = 0)
	count_error = abs(count_generated_means - count_train_means)
	mae_count = count_error.mean()

	binary_train_means = btrain.mean(axis = 0)
	binary_generated_means = bgenerated.mean(axis = 0)
	binary_error = abs(binary_generated_means - binary_train_means)
	mae_binary = binary_error.mean()

	df_train_error = pd.DataFrame()
	df_train_error['icd_codes'] = ch
	df_train_error['binary_train_mean_error'] = binary_train_means.tolist()
	df_train_error['count_train_mean_error'] = count_train_means.tolist()
	df_train_error['binary_generated_mean_error'] = binary_generated_means.tolist()
	df_train_error['count_generated_mean_error'] = count_generated_means.tolist()
	df_train_error['binary_absolute_error'] = binary_error
	df_train_error['count_absolute_error'] = count_error
	df_train_error.to_csv("../summary_stats/icd_codes_train_error_matrix.csv")

	df_mae = pd.DataFrame()
	df_mae['binary_mae'] = mae_binary
	df_mae['count_mae'] = mae_count
	df_mae['binary_no_record_columns'] = binary_generated_means[binary_generated_means == 0].shape
	df_mae['count_no_record_columns'] = count_generated_means[count_generated_means == 0].shape

	df_mae.to_csv("../summary_stats/mae_train.csv")


############Test
	
	ba_test = np.load("../generated/test_binary.matrix", allow_pickle = True)
	ca_test = np.load("../generated/test_count.matrix", allow_pickle = True)
	print(ba_test.shape)
	print(ca_test.shape)


	ctest = pd.DataFrame(data = ca_test,
						columns = ch)
	btest = pd.DataFrame(data = ba_test,
						columns = bh)

	cgenerated = pd.DataFrame(data = count_dat,
						columns = ch)
	bgenerated = pd.DataFrame(data = binary_dat,
						columns = bh)
	bgenerated[bgenerated >= 0.5] = 1
	bgenerated[bgenerated < 0.5] = 0

	count_test_means = ctest.mean(axis = 0)
	count_generated_means = cgenerated.mean(axis = 0)
	count_error = abs(count_generated_means - count_test_means)
	mae_count = count_error.mean()

	binary_test_means = btest.mean(axis = 0)
	binary_generated_means = bgenerated.mean(axis = 0)
	binary_error = abs(binary_generated_means - binary_test_means)
	mae_binary = binary_error.mean()

	df_test_error = pd.DataFrame()
	df_test_error['icd_codes'] = ch
	df_test_error['binary_test_mean_error'] = binary_test_means.tolist()
	df_test_error['count_test_mean_error'] = count_test_means.tolist()
	df_test_error['binary_generated_mean_error'] = binary_generated_means.tolist()
	df_test_error['count_generated_mean_error'] = count_generated_means.tolist()
	df_test_error['binary_absolute_error'] = binary_error
	df_test_error['count_absolute_error'] = count_error
	df_test_error.to_csv("../summary_stats/icd_codes_test_error_matrix.csv")

	df_mae = pd.DataFrame()
	df_mae['binary_mae'] = mae_binary
	df_mae['count_mae'] = mae_count
	df_mae['binary_no_record_columns'] = binary_generated_means[binary_generated_means == 0].shape
	df_mae['count_no_record_columns'] = count_generated_means[count_generated_means == 0].shape

	df_mae.to_csv("../summary_stats/mae_test.csv")


## Calculate correlations
	count_train_corr = ctrain.corr()
	binary_train_corr = btrain.corr()
	count_test_corr = ctest.corr()
	binary_test_corr = btest.corr()
	count_generated_corr = cgenerated.corr()
	binary_generated_corr = bgenerated.corr()
	
	count_train_corr.to_csv("../summary_stats/count_train_corr.csv")
	binary_train_corr.to_csv("../summary_stats/binary_train_corr.csv")
	count_test_corr.to_csv("../summary_stats/count_test_corr.csv")
	binary_test_corr.to_csv("../summary_stats/binary_test_corr.csv")
	count_generated_corr.to_csv("../summary_stats/count_generated_corr.csv")
	binary_generated_corr.to_csv("../summary_stats/binary_test_generated.csv")


	# tmp = binary_dat[1]
	# tmp = tmp[tmp >= 0.01]
	# print(tmp)
	# # print(binary_dat[0])
	# exit(0)
	# count_file = "../synthetic/cerner_count.npy" 
	# count_dat = np.load(count_file)
	# print(count_dat.shape)

	# print(count_dat)
