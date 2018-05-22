import DataHandler
import Model


def train_char(Mod,char,epoch):
    '''Trains the model a character for epoch number of times '''

    print("Character ",char )
    images,labels=DataHandler.load_char_data(char)

    for e in range(epoch):
        print("Epoch {}".format(e+1))
        accuracy=0
        loss=10
        prev_loss=-1
        loss_stag_count=0
        while(accuracy<95 or (loss > 2 and loss_stag_count < 50) ):
            loss,accuracy=Mod.train(images,labels)
            if(loss==prev_loss):
                loss_stag_count+=1
            prev_loss=loss

    return


def train(Mod,char_set,epoch):

    '''Trains the Model'''

    for char in char_set:
        train_char(Mod,char,epoch)

    return


def trainer():
    '''This module trains the Network adn Dumps it into the Disk'''
    Mod=Model.Model()
    Mod.construct()
    Mod.restore()

    train(Mod,char_set=[1,2,1,2,1,2],epoch=5)

    Mod.save()


trainer()
