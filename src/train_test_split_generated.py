import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def convertTo3DigitIcd9(dxStr):
	return str(dxStr)[:3]	


def trainTestSplitEncounter(adm_file, diag_file, t_size = 0.2, common = 1):
	adm = pd.read_csv(adm_file)
	dia = pd.read_csv(diag_file)

	dia['icd_3digit'] = dia['icd_codes'].map(convertTo3DigitIcd9)
	print(len(dia['icd_3digit'].unique()))


	adm['pid-vid'] = adm['pid'].apply(lambda x: str(x)) + "-" + adm['visit_id'].apply(lambda x: str(x))
	dia['pid-vid'] = dia['pid'].apply(lambda x: str(x)) + "-" + dia['visit_id'].apply(lambda x: str(x))
	
	#combining pid check
	apvid = adm['pid-vid'].tolist()
	dpvid = dia['pid-vid'].tolist()
	cpvid = list(set(apvid) & set(dpvid))
	adm = adm[adm['pid-vid'].isin(cpvid)]
	dia = dia[dia['pid-vid'].isin(cpvid)]

	# print(adm[adm['visit_id'] == 2014])
	# exit(0)

	adm_train, adm_test = train_test_split(adm, test_size = t_size)
	apv_train = adm_train['pid-vid'].tolist()
	apv_test = adm_test['pid-vid'].tolist()

	dia_train = dia[dia['pid-vid'].isin(apv_train)]
	dia_test = dia[dia['pid-vid'].isin(apv_test)]

	adm_train = adm_train.drop(['pid-vid'], axis = 1)
	adm_test = adm_test.drop(['pid-vid'], axis = 1)
	dia_train = dia_train.drop(['pid-vid'], axis = 1)
	dia_test = dia_test.drop(['pid-vid'], axis = 1)

	dia_train['icd_3digit'] = dia_train['icd_codes'].map(convertTo3DigitIcd9)

	dia_test['icd_3digit'] = dia_test['icd_codes'].map(convertTo3DigitIcd9)

	if common == 1:
		train_icd = dia_train['icd_3digit']
		test_icd = dia_test['icd_3digit']
		common_icd = list(set(train_icd).intersection(set(test_icd)))
		dia_train = dia_train.loc[dia_train['icd_3digit'].isin(common_icd)]
		dia_test = dia_test.loc[dia_test['icd_3digit'].isin(common_icd)]


	print(len(dia_train['icd_3digit'].unique()))
	print(len(dia_test['icd_3digit'].unique()))


	dia = dia.drop(['icd_3digit'], axis = 1)
	dia_train = dia_train.drop(['icd_3digit'], axis = 1)
	dia_test = dia_test.drop(['icd_3digit'], axis = 1)


	# print(adm_train[adm_train['visit_id'].isin(['2014'])])
	# print(dia[dia['visit_id'].isin(['2014'])])
	# exit(0)

	# print(adm_train.shape)
	# print(adm_test.shape)
	# print(dia_train.shape)
	# print(dia_test.shape)

	return adm_train, dia_train, adm_test, dia_test


if __name__ == "__main__":

	t_size = 0.2

	af = "../generated/ADMISSIONS_GENERATED_NOMERGE.csv"
	df = "../generated/DIAGNOSES_ICD_GENERATED_NOMERGE.csv"
	adm_train, dia_train, adm_test, dia_test = trainTestSplitEncounter(af, df, t_size = t_size, common = 1)
	
	print("Shape of ADMISSION TRAIN:", adm_train.shape)
	print("Shape of ADMISSION TEST:", adm_test.shape)
	print("Shape of DIA TRAIN:", dia_train.shape)
	print("Shape of DIA TEST:", dia_test.shape)


	adm_train.to_csv("../generated/ADMISSIONS_GENERATED_TRAIN.csv", index = False)
	dia_train.to_csv("../generated/DIAGNOSES_ICD_GENERATED_TRAIN.csv", index = False)
	adm_test.to_csv("../generated/ADMISSIONS_GENERATED_TEST.csv", index = False)
	dia_test.to_csv("../generated/DIAGNOSES_ICD_GENERATED_TEST.csv", index = False)

	
	# loading the binary dataset
	# adm_chk = pd.read_csv("../generated/ADMISSIONS_GENERATED_NOMERGE.csv")
	# dia_chk = pd.read_csv("../generated/DIAGNOSES_ICD_GENERATED_NOMERGE.csv")
	# chk = pd.merge(adm_chk, dia_chk, on = ["pid", "visit_id"], how = "outer")
	# chk['icd_3digit'] = chk['icd_codes'].map(convertTo3DigitIcd9)
	# print(len(chk['icd_3digit'].unique()))


	# adm = pd.read_csv("../generated/ADMISSIONS_GENERATED.csv")
	# dia = pd.read_csv("../generated/DIAGNOSES_ICD_GENERATED.csv")

	# orig = pd.merge(adm, dia, on = ["pid", "visit_id"], how = "inner")
	# orig['icd_3digit'] = orig['icd_codes'].map(convertTo3DigitIcd9)
	# print(len(orig['icd_3digit'].unique()))




	# icd_headers_binary_sumstats = np.load("../pretrain/binary.types", allow_pickle = True)
	# icd_headers_binary = [i for i in icd_headers_binary_sumstats.keys()]
	# # print(len(icd_headers_binary))
	# print(icd_headers_binary)


	# # raw = np.load("../synthetic/binary/cerner_synthetic.npy")
	# # print(raw.shape)
	# # print(raw[10])

	# ## Analysis for count
	# raw = np.load("../synthetic/count/cerner_synthetic.npy")
	# print(raw.shape)

	
	# print(raw[0])