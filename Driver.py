import DataHandler
import Model
import tools
from conf import *
label_set=[ i for i in range(12) ]

def load_data(filter=False):
    '''Loads and Filters the Data'''
    images,labels=DataHandler.load_user_data()
    if(filter):
        images,labels=tools.FilterData(images,labels,label_set)
    return images,labels


def load_data_Users(usernos,filter=False):
    images=[]
    labels=[]

    for _ in range(usernos):
        images_,labels_=load_data(filter)
        images.extend(images_)
        labels.extend(labels_)

    print("Size of current epoch dataset",len(labels))
    return images,labels



def train_user(Mod,epoch,usernos):
    images,labels=load_data_Users(filter=False,usernos)
    for e in range(epoch):
        print("Epoch {}".format(e))
        loss=0
        prev_loss=-1
        acc=0
        loss_stagnant=0

        while(acc<95 and  loss_stagnant<10):

            loss,acc=Mod.train(images,labels)
            if(prev_loss==loss):
                loss_stagnant+=1
                print("="*30)
            else:
                loss_stagnant=0
                prev_loss=loss

        if(loss_stagnant==10):
            print("Loss Stagnated at ",prev_loss)

    return

def train(Mod,epoch,usernos):
    '''Trains the model for a usernos no of times '''

    for _ in range(usernos):
        train_user(Mod,epoch,usernos)

    return


def trainer(restore):
    '''This module trains the Network adn Dumps it into the Disk'''
    Mod=Model.Model()
    Mod.construct()
    if(restore):
        Mod.restore()

    train(Mod,epoch=1,usernos=15)

    Mod.save()

restore=False

if(__name__=="__main__"):
    trainer(restore)
