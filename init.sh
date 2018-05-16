#!/usr/bin/bash
#This scripts executes initiazlization steps- Extracts the archieves and Rearranges them


if [ -d Dataset ]; then
	echo "Direcory Exists"
else
	echo "Extracting Files"
	tar -xf Datasets/hpl-tamil-iso-char-offline-1.0.tar.gz -C .
	echo "Rearranging Files"
	python RearrangeData.py
	echo "Rearrange Done"
fi
if [ -d tamil_dataset_offline ];then 

	echo "Removing tamil_dataset_offline"
	rm -r tamil_dataset_offline
	echo "Removal Complete"
fi
