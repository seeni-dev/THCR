'''
    This module is re arrange the dataset tamil_dataset_offline into Dataset folder which organizes data by labels
'''

import os
import sys
from shutil import copyfile
import matplotlib.pyplot as plt

root_source='tamil_dataset_offline'
root_destination='Dataset'

os.makedirs(root_destination,exist_ok=True)

counter={}

def invalidLabel(label):
    '''label is a string'''
    try:
        label_i=int(label) #Exception will be raised if it is not a character label
        return False
    except:
        return True

def copyIfApplicable(file_source,image_name):
    '''This method copies the files only if it is image file. It also organizes the images into different directories'''
    try:
        label=image_name[:3]
        if(invalidLabel(label)):
            return
        #incremeent the counter
        try:
            counter[label]+=1
        except:
            counter[label]=1
    except:
        print("Invalid File")

    dir_desination=root_destination+"/"+label+"/"
    os.makedirs(dir_desination,exist_ok=True)
    file_destination=dir_desination+str(counter[label])
    copyfile(file_source,file_destination)
    return

for user_dir in os.listdir(root_source):
    for image_name in os.listdir(root_source+"/"+user_dir):
        file_source=root_source+"/"+user_dir+"/"+image_name
        copyIfApplicable(file_source,image_name)


character=counter.keys()
counts_character=[counter[key] for key in character]
plt.scatter(character,counts_character)
plt.show()
