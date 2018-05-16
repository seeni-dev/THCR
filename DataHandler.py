import pickle
import os
import sys
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np
import tools

image_size=100
num_characters=156

def getImageData(file_path):
    '''This method will read the image and resize according to the size (image_size,image_size)'''
    image=plt.imread(file_path)
    image_corrected=rgb2gray(image)
    image_resized=resize(image_corrected,output_shape=[image_size,image_size])
    return image_resized

def onehot(label):
    onehot_label=np.zeros((num_characters),np.uint8)
    onehot_label[int(label)]=1
    return onehot_label

def makePickles():
    '''This module create the pickels for all the files for each chracter'''

    root_source="Dataset"
    root_destination="Pickles"
    os.makedirs(root_destination,exist_ok=True)

    for character in os.listdir(root_source):
        dir_source=root_source+"/"+character
        dir_destination_pickle=root_destination+"/"+character+".pkl"
        images=[]
        labels=[]
        for image in os.listdir(dir_source):
            file_source=dir_source+"/"+image
            try:
                image=getImageData(file_source)
                images.append(image)
                label=onehot(character)
                labels.append(label)
            except:
                print("Error in proceeing file ",file_source)

        save={
            "images":images,
            "labels":labels
        }
        with open(dir_destination_pickle,'wb') as file_p:
            pickle.dump(save,file_p)

char_count=0
def load_train_data():
    global char_count
    char_count%=156
    char=tools.make_char_labe_from_int(char_count)
    char_count+=1
    pickle_file="Pickles/"+char+".pkl"
    pickle_file=open(pickle_file,"rb")
    save=pickle.load(pickle_file)
    print("Train character:",char)
    return save["images"],save["labels"]
