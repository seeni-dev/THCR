import tensorflow as tf
class Model():
    def __init__(self):
        print("Model Initializing")

    def construct(self):
        ''' It constructs the model'''
        tf.reset_default_graph()
        self.image_size=100
        self.num_characters=156
        self.image=tf.placeholder(tf.float32,shape=[None,self.image_size,self.image_size],name="image")
        self.label=tf.placeholder(tf.float32,shape=[None,self.num_characters],name="label")

        self.layer0=tf.reshape(self.image,[-1,self.image_size,self.image_size,1],name="layer0")

        self.layer1=tf.layers.conv2d(self.layer0,filters=32,kernel_size=[4,4],strides=[2,2],activation=tf.nn.relu)

        self.layer2=tf.layers.conv2d(self.layer1,filters=64,kernel_size=[2,2],strides=[2,2],activation=tf.nn.relu)

        self.layer3=tf.layers.conv2d(self.layer2,filters=128,kernel_size=[2,2],strides=[2,2],activation=tf.nn.relu)

        self.layer4=tf.layers.conv2d(self.layer3,filters=256,kernel_size=[2,2],strides=[2,2],activation=tf.nn.relu)

        self.layer5=tf.layers.conv2d(self.layer4,filters=512,kernel_size=[2,2],strides=[2,2],activation=tf.nn.relu)

        self.layer6=tf.layers.flatten(self.layer5)

        self.layer7=tf.layers.dense(self.layer6,units=self.num_characters,activation=tf.nn.sigmoid,name="layer7")


        self.logits=tf.nn.softmax(self.layer7,name="logits")


        self.loss=tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=self.label,logits=self.logits))

        self.accuracy=tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.argmax(self.label,1),tf.argmax(self.logits,1)
                ),tf.float32
            )
        )*100;

        self.optimizer=tf.train.GradientDescentOptimizer(0.5).minimize(self.loss)

        self.sess=tf.InteractiveSession()
        tf.global_variables_initializer().run()

    def train(self,images,labels):
        '''Trainer for the network'''
        _,lo,acc=self.sess.run([self.optimizer,self.loss,self.accuracy],feed_dict={self.image:images,self.label:labels})
        print(lo,acc)
        if(acc==100):
            return 1
        return 0

    def predict(self,image):
        '''predictor'''
        prediction=self.sess.run([self.logits],feed_dict={self.image:image})
        print(prediction)

    def save(self):
        saver=tf.train.Saver()
        saver.save(self.sess,"DUMP/Model.ckpt")
        print("Model Saved")

    def restore(self):
        saver=tf.train.Saver()
        saver.restore(sess=self.sess,save_path="DUMP/Model.ckpt")
        print("Model Restored")
