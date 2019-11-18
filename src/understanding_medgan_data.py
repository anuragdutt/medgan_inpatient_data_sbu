import h5py
import numpy as np
from datetime import datetime
import pandas as pd



if __name__ == "__main__":

	filename = "../data/hf_143_inpatient_50_plus_1lab_1dx_combined.hdf5"
	f = h5py.File(filename, 'r')


	ohd = f['ohdsi']

	# patient id field
	pid_header = ohd['identifiers/person/column_annotations']
	pid = ohd['identifiers/person/core_array']

	# visit occurences
	visit_header = ohd['visit_occurrence/column_annotations']
	visit = ohd['visit_occurrence/core_array']

	## visit_id
	visit_id_header = ohd['identifiers/visit_occurrence/column_annotations']
	visit_id = ohd['identifiers/visit_occurrence/core_array']

	# ICD codes:
	icd_keys = ohd["condition_occurrence/source_concept/"]


	icd_header_annon = ohd["condition_occurrence/source_concept/column_annotations"]
	icd = ohd["condition_occurrence/source_concept/core_array"]
	icd_codes = np.array([code.decode('UTF-8') for code in icd_header_annon[2,]])

	



	visit_ids = []
	pids = []
	visit_start = []
	visit_end = []


	diag_col = ["pid", "visit_id", "seq", "icd_codes"]
	diagnosis_icd = pd.DataFrame(columns = diag_col)

	# for i in range(icd.shape[0]):
	for i in range(10001	):
		vid = int(visit_id[i,0])
		pat = int(pid[i,0])
		vs = datetime.utcfromtimestamp(int(visit[i,7])).strftime('%Y-%m-%d %H:%M:%S')
		ve = datetime.utcfromtimestamp(int(visit[i,8])).strftime('%Y-%m-%d %H:%M:%S')
		
		visit_ids.append(vid)
		pids.append(pat)
		visit_start.append(vs)
		visit_end.append(ve)	
		

		icd9 = icd_codes[np.where(icd[i,:] != 0)]
		num_codes = icd9.size
		parr = np.repeat(pat, num_codes)
		varr = np.repeat(vid, num_codes)
		seq = np.arange(num_codes) + 1
		app = pd.DataFrame(list(map(list, zip(*[parr.tolist(), varr.tolist(), seq.tolist(), icd9.tolist()]))), columns = diag_col)
		diagnosis_icd = diagnosis_icd.append(app)
		if i % 1000 == 0:
			print("Completed: " + str(i))





	admissions = pd.DataFrame({"pid" : pids,
								"visit_id": visit_ids,
								"visit_start": visit_start,
								"visit_end": visit_end})


	admissions.to_csv("../generated/ADMISSIONS_GENERATED_NOMERGE.csv", index = False)
	diagnosis_icd.to_csv("../generated/DIAGNOSES_ICD_GENERATED_NOMERGE.csv", index = False)




	adm = pd.read_csv("../generated/ADMISSIONS_GENERATED_NOMERGE.csv")
	dia = pd.read_csv("../generated/DIAGNOSES_ICD_GENERATED_NOMERGE.csv")

	# print(adm.shape)
	# print(dia.shape)

	# df = pd.merge(adm, dia, on = ["pid", "visit_id"], how = "inner")

	# adm = df.loc[:,['pid', 'visit_id', 'visit_start', 'visit_end']]
	# adm['pid-vid'] = adm['pid'] + adm['visit_id']

	# dia = df.loc[:, ['pid', 'visit_id', 'seq', 'icd_codes']]

	# adm.to_csv("../generated/ADMISSIONS_GENERATED.csv", index = False)
	# dia.to_csv("../generated/DIAGNOSES_ICD_GENERATED.csv", index = False)


