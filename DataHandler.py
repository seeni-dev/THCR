import pickle
import os
import sys
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np
import tools
from RearrangeData import invalidLabel

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



def data_from_path(path,threshold_size=-1):
    ''' given path to the pickle file load it into memory '''

    with open(path,'rb') as pickle_file:

        save=pickle.load(pickle_file)
        if(threshold_size>=len(save["images"]) or threshold_size==-1):
            return save["images"],save["labels"]
        else:
            return save["images"][:threshold_size],save["labels"][:threshold_size]


def mix_character(characters,threshold_size=-1):
    ''' mix the character symbols with each character having threshold_size'''

    characters=[tools.make_char_labe_from_int(c) for c in characters]

    images=[]
    labels=[]

    for char in characters:
        pickle_path="Pickles/"+char+".pkl"
        images_,labels_=data_from_path(pickle_path,threshold_size=-1)

        images.extend(images_)
        labels.extend(labels_)

    return images,labels


def load_char_data(char):
    '''returns the image data for the char'''

    char_=tools.make_char_labe_from_int(char)
    pickle_path="Pickles/"+char_+".pkl"

    return data_from_path(pickle_path)



def makePickle_Users():

    ''' Make Pickles based on the user wise data not character_wise data '''

    root_source="tamil_dataset_offline"
    root_destination="Pickles_User"

    os.makedirs(root_destination,exist_ok=True)

    for user in os.listdir(root_source):
        directory_source=os.path.join(root_source,user)
        pickle_destination=os.path.join(root_destination,user)+".pkl"
        images=[]
        labels=[]

        for file in os.listdir(directory_source):

            file_source=os.path.join(directory_source,file)
            try:
                image=getImageData(file_source)
                label=file[:3]
                if(invalidLabel(label)):
                    print("Invalid File",file_source)
                    return
                label=onehot(label)
                images.append(image)
                labels.append(label)

            except:
                print("Invalid File",file_source)

        #when all the files added pickle it
        save={
            "images":images,
            "labels":labels
        }

        with open(pickle_destination,"wb") as pickle_file:
            pickle.dump(file=pickle_file,obj=save)

    return



if(__name__=="__main__"):
    makePickle_Users()
