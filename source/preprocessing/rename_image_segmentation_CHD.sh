input_img_directory='/home/axel/dev/fetus_GA_prediction/data/raw/MRI/chd_fetuses'
input_seg_directory='/home/axel/dev/fetus_GA_prediction/data/raw/delineation/chd_fetuses'
output_img_directory='/home/axel/dev/fetus_GA_prediction/data/raw_renamed/MRI/chd_fetuses'
output_seg_directory='/home/axel/dev/fetus_GA_prediction/data/raw_renamed/delineation/chd_fetuses'

subject_id_file='/home/axel/dev/fetus_GA_prediction/data/CHD_fetus_ID_with_GA.csv'

while IFS=, read  row study fetus_ID scan scan_info	GA Notes
#while IFS=, read  row study fetus_ID scan scan_info
do 
		
	if [[ $study == "abc" ]]
	then 
# 		cp ${input_img_directory}/${study}/${study}_${fetus_ID}_0${scan}*.nii.gz ${output_img_directory}/${study}/${study}_${fetus_ID}_scan_0${scan}.nii.gz
# 		cp ${input_img_directory}/${study}/${study}_${fetus_ID}_scan_0${scan}*.nii.gz ${output_img_directory}/${study}/${study}_${fetus_ID}_scan_0${scan}.nii.gz
# 	
# 		cp ${input_seg_directory}/${study}/${study}_${fetus_ID}_0${scan}*.nii.gz ${output_seg_directory}/${study}/${study}_${fetus_ID}_scan_0${scan}_delineation.nii.gz
# 		cp ${input_seg_directory}/${study}/${study}_${fetus_ID}_scan_0${scan}*.nii.gz ${output_seg_directory}/${study}/${study}_${fetus_ID}_scan_0${scan}_delineation.nii.gz
		
		cp ${input_img_directory}/${study}/fetus_${fetus_ID}_0${scan}*.nii.gz ${output_img_directory}/${study}/${study}_${fetus_ID}_scan_0${scan}.nii.gz
		cp ${input_img_directory}/${study}/fetus_${fetus_ID}_scan_0${scan}*.nii.gz ${output_img_directory}/${study}/${study}_${fetus_ID}_scan_0${scan}.nii.gz
	
		cp ${input_seg_directory}/${study}/fetus_${fetus_ID}_0${scan}*.nii.gz ${output_seg_directory}/${study}/${study}_${fetus_ID}_scan_0${scan}_delineation.nii.gz
		cp ${input_seg_directory}/${study}/fetus_${fetus_ID}_scan_0${scan}*.nii.gz ${output_seg_directory}/${study}/${study}_${fetus_ID}_scan_0${scan}_delineation.nii.gz
	
	fi
	
done < $subject_id_file