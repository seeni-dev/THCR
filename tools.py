from conf import *

def make_char_labe_from_int(label):
    if(label>=num_characters):
        raise Exception("Label Out of Range or Invalid File ")
    char_label=str(label)
    while(len(char_label)<3):
        char_label="0"+char_label
    return char_label

def argmax(one_hot_label):
    ''' this method converts the one hot label into a index. This is implemented for ease of use '''
    label_i=0
    for i in range(len(one_hot_label)):
        if(one_hot_label[i]):
            label_i=i
            return label_i
    raise Exception("Invalid One Hot label. Check the Dimesions")


def invalidLabel(label):
    '''label is a string'''
    try:
        label_i=int(label) #Exception will be raised if it is not a character label
        if(label_i>=num_characters):
            return True
        return False
    except:
        return True



def FilterData(images,labels,label_set):
    images_f=[]
    labels_f=[]
    for label_i in range(len(labels)):
        label=labels[label_i]
        if(argmax(label) in label_set):
            images_f.append(images[label_i])
            labels_f.append(label)
    return images_f,labels_f
