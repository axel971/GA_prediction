
input_img_directory='/home/axel/dev/fetus_GA_prediction/data/raw_renamed/MRI/chd_fetuses/all'
input_seg_directory='/home/axel/dev/fetus_GA_prediction/data/raw_renamed/delineation/chd_fetuses/all'

output_seg_largestConnectedComponent_directory='/home/axel/dev/fetus_GA_prediction/data/preprocessing/delineation/CHD/largestConnectedComponenent'

output_img_boundingBox_directory='/home/axel/dev/fetus_GA_prediction/data/preprocessing/MRI/CHD/boundingBox'
output_seg_boundingBox_directory='/home/axel/dev/fetus_GA_prediction/data/preprocessing/delineation/CHD/boundingBox'

output_img_resampledBoundingBox_directory='/home/axel/dev/fetus_GA_prediction/data/preprocessing/MRI/CHD/resampled_boundingBox'
output_seg_resampledBoundingBox_directory='/home/axel/dev/fetus_GA_prediction/data/preprocessing/delineation/CHD/resampled_boundingBox'

output_img_histogramMatching="/home/axel/dev/fetus_GA_prediction/data/preprocessing/MRI/CHD/histogramMatching"

subject_id_file='/home/axel/dev/fetus_GA_prediction/data/CHD_fetus_ID_with_GA.csv'


while IFS=, read  row study fetus_ID scan scan_info	GA Notes
do 
		
	if [[ $study != "study" ]]
	then 
	
		echo "Delineation preprocessing (largest connected component extraction + dilatation performed on the masks): " ${study}_${fetus_ID}_scan_0${scan}
		./extractLargestConnectedComponent/build/extractLargestConnectedComponent ${input_seg_directory}/${study}_${fetus_ID}_scan_0${scan}*.nii.gz ${output_seg_largestConnectedComponent_directory}/${study}_${fetus_ID}_scan_0${scan}.nii.gz 
		./dilatation/build/dilatation ${output_seg_largestConnectedComponent_directory}/${study}_${fetus_ID}_scan_0${scan}.nii.gz ${output_seg_largestConnectedComponent_directory}/${study}_${fetus_ID}_scan_0${scan}.nii.gz 	
		
		echo "Bounding box calculation and extraction: " ${study}_${fetus_ID}_scan_0${scan}		
		./boundingBox/build/boundingBox ${input_img_directory}/${study}_${fetus_ID}_scan_0${scan}.nii.gz ${output_seg_largestConnectedComponent_directory}/${study}_${fetus_ID}_scan_0${scan}.nii.gz ${output_img_boundingBox_directory}/${study}_${fetus_ID}_scan_0${scan}.nii.gz 
		./boundingBox/build/boundingBox ${output_seg_largestConnectedComponent_directory}/${study}_${fetus_ID}_scan_0${scan}.nii.gz ${output_seg_largestConnectedComponent_directory}/${study}_${fetus_ID}_scan_0${scan}.nii.gz ${output_seg_boundingBox_directory}/${study}_${fetus_ID}_scan_0${scan}.nii.gz 
	
		echo "Resampling of the Bounding boxes: " ${study}_${fetus_ID}_scan_0${scan}		
		./resampling/build/resampling "128" "128" "128" ${output_img_boundingBox_directory}/${study}_${fetus_ID}_scan_0${scan}.nii.gz ${output_img_resampledBoundingBox_directory}/${study}_${fetus_ID}_scan_0${scan}.nii.gz 
		./resampling_mask/build/resampling_mask "128" "128" "128" ${output_seg_boundingBox_directory}/${study}_${fetus_ID}_scan_0${scan}.nii.gz ${output_seg_resampledBoundingBox_directory}/${study}_${fetus_ID}_scan_0${scan}.nii.gz 
	
	fi
	
done < $subject_id_file

referenceImage="/home/axel/dev/fetus_GA_prediction/data/preprocessing/MRI/resampled_boundingBox/abc_00579_scan_01.nii.gz"
while IFS=, read  row study fetus_ID scan scan_info	GA Notes
do 
		
	if [[ $study != "study" ]]
	then 
	
		echo "Histogram matching: " ${study}_${fetus_ID}_scan_0${scan}
		./histogram_matching/build/histogramMatching ${output_img_resampledBoundingBox_directory}/${study}_${fetus_ID}_scan_0${scan}.nii.gz ${referenceImage} ${output_img_histogramMatching}/${study}_${fetus_ID}_scan_0${scan}.nii
	
	fi
	
done < $subject_id_file