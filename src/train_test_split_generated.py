import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def convertTo3DigitIcd9(dxStr):
	return str(dxStr)[:3]	


def trainTestSplit(adm_file, diag_file, t_size = 0.2):
	adm = pd.read_csv(adm_file)
	dia = pd.read_csv(diag_file)


	dat = pd.merge(adm, dia, on = ["pid", "visit_id"], how = "inner")
	print(adm.shape)
	print(dia.shape)
	print(dat.shape)
	train, test = train_test_split(dat, test_size = t_size)

	adm_train = train.loc[:,['pid', 'visit_id', 'visit_start', 'visit_end']]
	dia_train = train.loc[:, ['pid', 'visit_id', 'seq', 'icd_codes']]

	adm_test = test.loc[:,['pid', 'visit_id', 'visit_start', 'visit_end']]
	dia_test = test.loc[:, ['pid', 'visit_id', 'seq', 'icd_codes']]

	return adm_train, dia_train, adm_test, dia_test


if __name__ == "__main__":

	# loading the binary dataset
	adm_chk = pd.read_csv("../generated/ADMISSIONS_GENERATED_NOMERGE.csv")
	dia_chk = pd.read_csv("../generated/DIAGNOSES_ICD_GENERATED_NOMERGE.csv")
	chk = pd.merge(adm_chk, dia_chk, on = ["pid", "visit_id"], how = "outer")
	chk['icd_3digit'] = chk['icd_codes'].map(convertTo3DigitIcd9)
	print(len(chk['icd_3digit'].unique()))



	# adm = pd.read_csv("../generated/ADMISSIONS_GENERATED.csv")
	# dia = pd.read_csv("../generated/DIAGNOSES_ICD_GENERATED.csv")

	# orig = pd.merge(adm, dia, on = ["pid", "visit_id"], how = "inner")
	# orig['icd_3digit'] = orig['icd_codes'].map(convertTo3DigitIcd9)
	# print(len(orig['icd_3digit'].unique()))

	t_size = 0.2

	af = "../generated/ADMISSIONS_GENERATED_NOMERGE.csv"
	df = "../generated/DIAGNOSES_ICD_GENERATED_NOMERGE.csv"
	adm_train, dia_train, adm_test, dia_test = trainTestSplit(af, df, t_size = t_size)
	

	print(adm_train.shape)
	print(adm_test.shape)
	exit(0)

	adm_train.to_csv("../generated/ADMISSIONS_GENERATED_TRAIN.csv", index = False)
	dia_train.to_csv("../generated/DIAGNOSES_ICD_GENERATED_TRAIN.csv", index = False)
	adm_test.to_csv("../generated/ADMISSIONS_GENERATED_TRAIN_TEST.csv", index = False)
	dia_test.to_csv("../generated/DIAGNOSES_ICD_GENERATED_TEST.csv", index = False)



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