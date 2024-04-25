
import sys
sys.path.append('..')

import numpy as np
import os
import pandas
import openpyxl

# Get the image quality label inside the xlsx file
file_dir = '/home/axel/dev/fetus_GA_prediction/data/new_subject_IDs.xlsx'
data = pandas.read_excel(file_dir)
data_length = data.shape[0]


data_study_name = []
data_fetus_ID = []
data_scan1 = []
data_scan1_info = []
data_scan2 = []
data_scan2_info =[]
data_scan3 = []
data_scan3_info =[]

output_study_name = []
output_fetus_ID = []
output_scan = []
output_scan_info = []


for n in range(data_length):
    data_study_name.append(data.loc[n].Study)
    data_fetus_ID.append(data.loc[n].Study_ID)
    data_scan1.append(data.loc[n].scan1)
    data_scan1_info.append(data.loc[n].scan1_info)
    data_scan2.append(data.loc[n].scan2)
    data_scan2_info.append(data.loc[n].scan2_info)
#     data_scan3.append(data.loc[n].scan3)
#     data_scan3_info.append(data.loc[n].scan3_info)

for n in range(data_length):
	if (data_scan1[n] == 1 and ((data_scan1_info[n] == "tbv") or (data_scan1_info[n] == "alllabelsconfirm")) ):
		output_study_name.append(data_study_name[n])
		output_fetus_ID.append(data_fetus_ID[n])
		output_scan.append(1)
		output_scan_info.append(data_scan1_info[n])
	
	if (data_scan2[n] == 1 and ((data_scan2_info[n] == "tbv") or (data_scan2_info[n] == "alllabelsconfirm"))):
		output_study_name.append(data_study_name[n])
		output_fetus_ID.append(data_fetus_ID[n])
		output_scan.append(2)
		output_scan_info.append(data_scan2_info[n])
# 	
# 	if (data_scan3[n] == 1 and ((data_scan3_info[n] == "tbv") or (data_scan3_info[n] == "alllabelsconfirm"))):
# 		output_study_name.append(data_study_name[n])
# 		output_fetus_ID.append(data_fetus_ID[n])
# 		output_scan.append(3)
# 		output_scan_info.append(data_scan3_info[n])
# 		
		
output_file_dir = '/home/axel/dev/fetus_GA_prediction/data/file_names/new_healphy_subjects_IDs.xlsx'
col1 = np.array(output_study_name).reshape(len(output_study_name), 1)
col2 = np.array(output_fetus_ID).reshape(len(output_fetus_ID),1)
col3 = np.array(output_scan).reshape(len(output_scan), 1)
col4 = np.array(output_scan_info).reshape(len(output_scan_info),1)

output_data = np.concatenate((col1, col2, col3, col4), axis = 1)
outputFile = pandas.DataFrame(output_data, columns=['study', 'fetus_ID', 'scan', 'scan_info'])
outputFile.to_excel(output_file_dir)

