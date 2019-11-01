import h5py
import numpy as np
from datetime import datetime
import pandas as pd



if __name__ == "__main__":

	filename = "../data/hf_143_inpatient_50_plus_1lab_1dx_combined.hdf5"
	f = h5py.File(filename, 'r')

	# for k in f.keys():
	# 	print(k)
	# exit(0)


	ohd = f['ohdsi']
	# for k in ohd.keys():
		# print(k)

	# ic9_header = ohd['condition_occurrence/column_annotations']
	# ic9 = ohd['condition_occurrence/core_array']
	# print(ic9_header[:4])
	# print(ic9.shape)
	# exit(0)

	# patient id field
	pid_header = ohd['identifiers/person/column_annotations']
	pid = ohd['identifiers/person/core_array']

	# visit occurences
	visit_header = ohd['visit_occurrence/column_annotations']
	visit = ohd['visit_occurrence/core_array']
	# print(visit_header.shape)
	# print(visit_header[:10])
	# exit(0)
	## visit_id
	visit_id_header = ohd['identifiers/visit_occurrence/column_annotations']
	visit_id = ohd['identifiers/visit_occurrence/core_array']
	# print(visit_id.shape)
	# print(visit_id_header[:4])

	# ICD codes:
	icd_keys = ohd["condition_occurrence/source_concept/"]


	icd_header_annon = ohd["condition_occurrence/source_concept/column_annotations"]
	icd = ohd["condition_occurrence/source_concept/core_array"]
	# print([code.decode('UTF-8') for code in icd_header_annon[2,]])
	# exit(0)
	# icd_codes = [np.frombuffer(code, dtype = "str")[0] for code in icd_header_annon[2,]]
	icd_codes = np.array([code.decode('UTF-8') for code in icd_header_annon[2,]])
	# print(icd_codes)
	# print(icd_header_annon[:4])
	# exit(0)


	visit_ids = []
	pids = []
	visit_start = []
	visit_end = []

	# for i in range(visit_id.shape[0]):
		# visit_ids.append(visit_id[i,0])
		# pids.append(pid[i,0])
		# visit_start.append(datetime.utcfromtimestamp(int(visit[i,7])).strftime('%Y-%m-%d %H:%M:%S'))
		# visit_end.append(datetime.utcfromtimestamp(int(visit[i,8])).strftime('%Y-%m-%d %H:%M:%S'))	

	diag_col = ["pid", "visit_id", "seq", "icd_codes"]
	diagnosis_icd = pd.DataFrame(columns = diag_col)

	# for i in range(icd.shape[0]):
	for i in range(10000):
		vid = int(visit_id[i,0])
		pat = int(pid[i,0])
		vs = datetime.utcfromtimestamp(int(visit[i,7])).strftime('%Y-%m-%d %H:%M:%S')
		ve = datetime.utcfromtimestamp(int(visit[i,8])).strftime('%Y-%m-%d %H:%M:%S')
		
		visit_ids.append(vid)
		pids.append(pat)
		visit_start.append(vs)
		visit_end.append(ve)	
		

		# for j in range(len(icd_codes)):
		# 	if icd[i,j] != 0:
		# 		seq = seq + 1
		# 		diagnosis_icd = diagnosis_icd.append({
		# 			"pid": int(pid[i, 0]), 
		# 			"visit_id": int(visit_id[i, 0]),
		# 			"seq": seq,
		# 			"icd_codes": icd_codes[j] 
		# 			}, ignore_index = True)

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


	admissions.to_csv("../generated/ADMISSIONS_GENERATED.csv", index = False)
	diagnosis_icd.to_csv("../generated/DIAGNOSES_ICD_GENERATED.csv", index = False)