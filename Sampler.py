import pickle
import matplotlib.pyplot as plt
from tools import argmax
def get_sample():
    char_label=input("Enter a Character Label(3 digits):")
    assert len(char_label)==3
    index=int(input("Enter a random index:"))
    pickle_file="Pickles/"+char_label+".pkl"
    pickle_file=open(pickle_file,"rb")
    save=pickle.load(pickle_file)
    try:
        image=save["images"][index]
        label=save["labels"][index]
    except:
        raise Exception("Index is so large and out of bounds")

    print("One hot label ",label)
    print("Argmax label",argmax(label))
    plt.imshow(image)
    plt.show()
    return image,label
