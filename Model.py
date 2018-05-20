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

        self.conv1=tf.layers.conv2d(self.layer0,32,kernel_size=[5,5],activation=tf.nn.relu)


        self.flat=tf.contrib.layers.flatten(self.layer0)

        self.dense0=tf.layers.dense(self.flat,units=1000,activation=tf.nn.sigmoid)

        self.dense1=tf.layers.dense(self.dense0,units=500,activation=tf.nn.sigmoid)

        self.dense=tf.layers.dense(self.dense1,units=self.num_characters,activation=tf.nn.sigmoid)

        self.logits=tf.nn.softmax(self.dense,name="logits")

        self.loss=tf.reduce_mean(tf.losses.softmax_cross_entropy (self.label,self.logits))

        self.accuracy=tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.argmax(self.label,1),tf.argmax(self.logits,1)
                ),tf.float32
            )
        )*100;

        self.optimizer=tf.train.GradientDescentOptimizer(1).minimize(self.loss)

        self.sess=tf.InteractiveSession()
        tf.global_variables_initializer().run()

        return

    def train(self,images,labels):
        '''Trainer for the network'''
        _,lo,acc=self.sess.run([self.optimizer,self.loss,self.accuracy],feed_dict={self.image:images,self.label:labels})
        print(lo,acc)
        return lo,acc

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
