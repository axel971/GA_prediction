

subject_id_file='/home/axel/dev/fetus_GA_prediction/data/CHD_fetus_ID_with_GA.csv'
subjectList='/home/axel/dev/fetus_GA_prediction/data/subjectList_CHD.csv'

while IFS=, read  row study fetus_ID scan scan_info	GA Notes
do 

	if [[ $study == "study" ]]
	then 
		
		echo "Subject_name"
		echo "Subject_name"";""GA" >> ${subjectList} 	
	fi
			
	if [[ $study != "study" ]]
	then 
		
		echo ${study}_${fetus_ID}_scan_0${scan}";"${GA}
		echo ${study}_${fetus_ID}_scan_0${scan}";"${GA} >> ${subjectList}	
	fi
	
done < $subject_id_file
