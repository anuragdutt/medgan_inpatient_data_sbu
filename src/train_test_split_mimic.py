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

	adm = adm.loc[:, ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME']]
	adm.columns = ['pid', 'visit_id', 'visit_start', 'visit_end']
	dia = dia.loc[:, ['SUBJECT_ID', 'HADM_ID', 'SEQ_NUM', 'ICD9_CODE']]
	dia.columns = ['pid', 'visit_id', 'seq', 'icd_codes']	

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

	return adm_train, dia_train, adm_test, dia_test


if __name__  == "__main__":

	t_size = 0.2

	af = "../mimic_raw/ADMISSIONS.csv"
	df = "../mimic_raw/DIAGNOSES_ICD.csv"

	adm_train, dia_train, adm_test, dia_test = trainTestSplitEncounter(af, df, t_size = t_size, common = 1)

	print("Shape of ADMISSION TRAIN:", adm_train.shape)
	print("Shape of ADMISSION TEST:", adm_test.shape)
	print("Shape of DIA TRAIN:", dia_train.shape)
	print("Shape of DIA TEST:", dia_test.shape)

	adm_train.to_csv("../mimic_raw/ADMISSIONS_TRAIN.csv", index = False)
	dia_train.to_csv("../mimic_raw/DIAGNOSES_ICD_TRAIN.csv", index = False)
	adm_test.to_csv("../mimic_raw/ADMISSIONS_TEST.csv", index = False)
	dia_test.to_csv("../mimic_raw/DIAGNOSES_ICD_TEST.csv", index = False)
