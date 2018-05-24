import DataHandler
import Model
import tools

label_set=[0,1,2,3,4,5,6,7,8,9,10,11,12]

def load_data(filter=False):
    '''Loads and Filters the Data'''
    images,labels=DataHandler.load_user_data()
    if(filter):
        images,labels=tools.FilterData(images,labels,label_set)
    return images,labels


def train_user(Mod,epoch):
    images,labels=load_data()
    for e in range(epoch):
        print("Epoch {}".format(e))
        loss=0
        prev_loss=-1
        acc=0
        loss_stagnant=0

        while(acc<95 and  loss_stagnant<30):

            loss,acc=Mod.train(images,labels)
            if(prev_loss==loss):
                loss_stagnant+=1
                print("="*30)
            else:
                loss_stagnant=0
                prev_loss=loss

        if(loss_stagnant==30):
            print("Loss Stagnated at ",prev_loss)

    return

def train(Mod,epoch,usernos):
    '''Trains the model for a usernos no of times '''

    for _ in range(usernos):
        train_user(Mod,epoch)

    return


def trainer(restore):
    '''This module trains the Network adn Dumps it into the Disk'''
    Mod=Model.Model()
    Mod.construct()
    if(restore):
        Mod.restore()

    train(Mod,epoch=1,usernos=150)

    Mod.save()

restore=False

if(__name__=="__main__"):
    trainer(restore)
