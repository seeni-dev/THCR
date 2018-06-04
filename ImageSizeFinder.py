from DataHandler import getImageData,setImageSize
import matplotlib.pyplot as plt
import os

image_sizes=[25,30,35,40,45,50]

def ShowImageSize(file_path,size):
    setImageSize(size)
    image=getImageData(file_path)
    plt.imshow(image)

def Driver():
    root_source="tamil_dataset_offline"

    user_dirs=os.listdir(root_source)

    #take the first user

    user=user_dirs[0]

    user_dir=os.path.join(root_source,user)

    for file in os.listdir(user_dir):
        file_path=os.path.join(user_dir,file)
        print(file_path)
        for size in image_sizes:
            print("Image Size ",size)
            ShowImageSize(file_path,size)

if(__name__=="__main__"):
    Driver()
