#!/usr/bin/bash
#This scripts executes initiazlization steps- Extracts the archieves and Rearranges them


if [ -d tamil_dataset_offline ]; then
	echo "Direcory Exists"
else
	#Download the files
	url=https://raw.githubusercontent.com/Seenivasanseeni/TamilCharacterRecognition/master/Datasets/hpl-tamil-iso-char-offline-1.0.tar.gz
	wget -P Datasets/ $url
fi

if [ -d tamil_dataset_offline ]; then

	echo "Extracting Files"
	tar -xf Datasets/hpl-tamil-iso-char-offline-1.0.tar.gz -C .
	echo "Extraction Done"

fi


#make pickles for al the image files
echo "Make Pickles start"
python DataHandler.py
echo "Make Pickles Done"

if [ -d tamil_dataset_offline ];then
	echo "Removing tamil_dataset_offline"
	rm -r tamil_dataset_offline
	echo "Removal Complete"
fi
