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


def getMae(original, generated, headers, binary = False):

	original = pd.DataFrame(data = original,
						columns = headers)
	generated = pd.DataFrame(data = generated,
						columns = headers)

	if binary:
		generated[generated >= 0.5] = 1
		generated[generated < 0.5] = 0

	
	original_means = original.mean(axis = 0)
	generated_means = generated.mean(axis = 0)
	merror = abs(generated_means - original_means)
	mae = merror.mean()


	df_error = pd.DataFrame()
	df_error['icd_codes'] = headers
	df_error['original_mean'] = original_means.tolist()
	df_error['generated_mean'] = generated_means.tolist()
	df_error['mean_error'] = merror

	count_null_columns = generated_means[generated_means == 0].shape

	return mae, df_error, count_null_columns

def getCor(mat, headers):
	dat = pd.DataFrame(mat, columns = headers)
	return pd.DataFrame(np.corrcoef(dat.values, rowvar = False), index = headers, columns = headers)


if __name__ == "__main__":

	# diag_icd_file_train = "../generated/DIAGNOSES_ICD_GENERATED_TRAIN.csv"
	# ba_train, ca_train = transformHeaders(diag_icd_file_train, ch)
	# pickle.dump(ba_train, open("../generated/train_binary.matrix", 'wb'), -1)
	# pickle.dump(ca_train, open("../generated/train_count.matrix", 'wb'), -1)
	

	# diag_icd_file_test = "../generated/DIAGNOSES_ICD_GENERATED_TEST.csv"
	# ba_test, ca_test = transformHeaders(diag_icd_file_test, ch)
	# pickle.dump(ba_test, open("../generated/test_binary.matrix", 'wb'), -1)
	# pickle.dump(ca_test, open("../generated/test_count.matrix", 'wb'), -1)

	count_headers = np.load("../pretrain/count.types", allow_pickle = True)
	binary_headers = np.load("../pretrain/binary.types", allow_pickle = True)

	ch = list(count_headers.keys())
	bh = list(binary_headers.keys())



	binary_file = "../synthetic/cerner_binary.npy" 
	binary_dat = np.load(binary_file)
	count_file = "../synthetic/cerner_count.npy" 
	count_dat = np.load(count_file)

########Train


	ba_train = np.load("../generated/train_binary.matrix", allow_pickle = True)
	ca_train = np.load("../generated/train_count.matrix", allow_pickle = True)

	mae_train_binary, df_train_binary, null_binary_train = getMae(ba_train, binary_dat, headers = ch, binary = True)
	mae_train_count, df_train_count, null_count_train = getMae(ca_train, count_dat, headers = ch, binary = False)

############Test

	ba_test = np.load("../generated/test_binary.matrix", allow_pickle = True)
	ca_test = np.load("../generated/test_count.matrix", allow_pickle = True)

	mae_test_binary, df_test_binary, null_binary_test = getMae(ba_test, binary_dat, headers = ch, binary = True)
	mae_test_count, df_test_count, null_count_test = getMae(ca_test, count_dat, headers = ch, binary = False)

	df_train_binary.to_csv("../summary_stats/cerner_absolute_errors_train_binary.csv", index = False)
	df_train_binary.to_csv("../summary_stats/cerner_absolute_errors_train_count.csv", index = False)
	df_test_binary.to_csv("../summary_stats/cerner_absolute_errors_test_binary.csv", index = False)
	df_test_binary.to_csv("../summary_stats/cerner_absolute_errors_test_count.csv", index = False)

	df_mae = pd.DataFrame([{'mae_train_count': mae_train_count,
							'mae_train_binary': mae_train_binary,
							'mae_test_count': mae_test_count,
							'mae_test_binary': mae_test_binary,
							'binary_null_columns': null_binary_train,
							'count_null_columns': null_count_train}
							])
	df_mae.to_csv("../summary_stats/cerner_mae.csv")

## Calculate correlations
	count_train_corr = getCor(ca_train, headers = ch)
	binary_train_corr = getCor(ca_train, headers = ch)
	count_test_corr = getCor(ca_test, headers = ch)
	binary_test_corr = getCor(ca_test, headers = ch)
	count_generated_corr = getCor(count_dat, headers = ch)
	binary_generated_corr = getCor(binary_dat, headers = ch)
	
	count_train_corr.to_csv("../summary_stats/count_train_corr.csv")
	binary_train_corr.to_csv("../summary_stats/binary_train_corr.csv")
	count_test_corr.to_csv("../summary_stats/count_test_corr.csv")
	binary_test_corr.to_csv("../summary_stats/binary_test_corr.csv")
	count_generated_corr.to_csv("../summary_stats/count_generated_corr.csv")
	binary_generated_corr.to_csv("../summary_stats/binary_test_generated.csv")


# x_train_Dataset

	x_train_original = np.load("../pretrain/x_train.matrix", allow_pickle = True)
	x_train_generated = np.load("../synthetic/x_train_binary_synthetic.npy")
	x_train_headers = np.load("../pretrain/x_train.types", allow_pickle = True)
	mae_x_train, df_x_train_errors, x_train_null = getMae(x_train_original, x_train_generated, x_train_headers, binary = True)
	x_train_original_corr = getCor(x_train_original, headers = x_train_headers) 
	x_train_generated_corr = getCor(x_train_generated, headers = x_train_headers) 
	print(mae_x_train)
	print(x_train_null)
	df_x_train_errors.to_csv("../summary_stats/x_train_absolute_errors.csv", index = False)
	x_train_original_corr.to_csv("../summary_stats/x_train_original_corr.csv")
	x_train_generated_corr.to_csv("../summary_stats/x_train_generated_corr.csv")

	# tmp = binary_dat[1]
	# tmp = tmp[tmp >= 0.01]
	# print(tmp)
	# # print(binary_dat[0])
	# exit(0)
	# count_file = "../synthetic/cerner_count.npy" 
	# count_dat = np.load(count_file)
	# print(count_dat.shape)

	# print(count_dat)
