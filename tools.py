def make_char_labe_from_int(label):
    if(label>=156):
        raise Exception("Invalid Label")
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
