import pickle
import os
import sys
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize
import numpy as np
import tools
from tools import invalidLabel

image_size=100
num_characters=156

def getImageData(file_path):
    '''This method will read the image and resize according to the size (image_size,image_size)'''
    image=plt.imread(file_path)
    image_corrected=rgb2gray(image)
    image_resized=resize(image_corrected,output_shape=[image_size,image_size])
    plt.imshow(image_resized)
    plt.show()
    return image_resized

def onehot(label):
    onehot_label=np.zeros((num_characters),np.uint8)
    onehot_label[int(label)]=1
    return onehot_label


def data_from_path(path,threshold_size=-1):
    ''' given path to the pickle file load it into memory '''

    with open(path,'rb') as pickle_file:

        save=pickle.load(pickle_file)
        if(threshold_size>=len(save["images"]) or threshold_size==-1):
            return save["images"],save["labels"]
        else:
            return save["images"][:threshold_size],save["labels"][:threshold_size]



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

root_source="Pickles_User"
user_pickles=os.listdir(root_source)
index=-1

def load_user_data():
    ''' Loads the user Data '''
    global index
    index=(index+1)%len(user_pickles)
    path=os.path.join(root_source,user_pickles[index])
    print("Using Pickle",path)
    return data_from_path(path)

def setImageSize(size):
    global image_size
    image_size=size



if(__name__=="__main__"):
    makePickle_Users()
