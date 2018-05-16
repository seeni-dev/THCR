import DataHandler
import Model

#if train
def train(Mod,epoch):
    for e in range(epoch):
        images,labels=DataHandler.load_train_data()
        print("Size",len(images))
        for _ in range(30):
            con_stat=Mod.train(images,labels)
            if(con_stat==1):
                break

epoch=100

def trainer():
    '''This module trains the Network adn Dumps it into the Disk'''
    Mod=Model.Model()
    Mod.construct()
    train(Mod,epoch)
    Mod.save()
    Mod="Dumped"
    Mod=Model.Model()
    Mod.construct()
    Mod.restore()
    DataHandler.char_count=0
    train(Mod,epoch)
    Mod.save()

trainer()
