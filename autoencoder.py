from matplotlib import pyplot as pp
from PIL import Image
import tensorflow as tf
import ctypes as ct
import numpy as np
import argparse,os,shutil,sys
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
    
#dataset
class Dataset:
    def __init__(self, path, mean):
        self.images=[]
        self.index=0
        for file in os.listdir(path):
            print('Reading: %s'%file)
            self.images.append(np.array(Image.open(path+'/'+file)))
            if mean:
                if not hasattr(self,'mean'):
                    self.mean=self.images[-1]
                else: self.mean+=self.images[-1]
        if mean:
            self.mean=np.divide(self.mean,float(len(self.images)))
            for image in self.images:
                image=np.subtract(image,self.mean)
        print('%d datums read!'%self.nrDatums())
    def nrDatums(self):
        return len(self.images)
    def nextBatch(self, sz, random):
        if random:
            arr=np.random.choice(self.nrDatums(),sz)
            return [self.images[i] for i in arr]
        else:
            id=self.index
            nr=self.nrDatums()
            self.index+=sz
            return [self.images[(i+id)%nr] for i in range(sz)]
    def nrEpoch(self):
        return self.index/self.nrDatums()
    def reset(self):
        self.index=0
#helper
def create_trainer(results, loss, fromss, toss, vars=None):
    vlist=[]
    if isinstance(vars,type(None)):
        vlist+=tf.trainable_variables()
    else:
        for name in vars:
            vlist+=tf.trainable_variables(name)
    #create minimizer
    with tf.variable_scope('optimizer',reuse=tf.AUTO_REUSE):
        global_step=tf.placeholder(tf.float32,name='global_step')
        alpha=pow(toss/fromss,1.0/float(results.max_epoch))
        learning_rate=tf.train.exponential_decay(fromss,global_step,1.0,alpha,staircase=True)
        trainStep=tf.train.AdamOptimizer(learning_rate).minimize(loss,var_list=vlist)
    print('Trainer info:')
    for v in vlist:
        print('\tVariable: %s!'%v.name)
    print('From stepsize %f to stepsize %f over %d epoch!'% \
          (learning_rate.eval(feed_dict={global_step:0}),learning_rate.eval(feed_dict={global_step:results.max_epoch}),results.max_epoch))
    return trainStep,learning_rate,global_step
#checkpoint
def load_latest(results, sess):
    latest_i=0
    saver=tf.train.Saver()
    for i in range(results.max_epoch):
        if i%results.save_interval == 0 and os.path.exists('%s/iter%d.ckpt.meta'%(results.model_path,i)):
            print('Restoring %s/iter%d.ckpt!'%(results.model_path,i))
            saver.restore(sess,'%s/iter%d.ckpt'%(results.model_path,i))
            latest_i=i+1
    return latest_i
def get_parser(model_default):
    parser=argparse.ArgumentParser(description='Train Network')
    parser.add_argument('--dataset_path',metavar='[path to dataset]',action='store',type=str,default='./dataset')
    parser.add_argument('--model_path',metavar='[path to saved model]',action='store',type=str,default=model_default)
    parser.add_argument('--graph_path',metavar='[path to saved model]',action='store',type=str,default='./graph')
    parser.add_argument('--max_epoch',metavar='[max epochs]',action='store',type=int,choices=range(0,1000),default=1000)
    parser.add_argument('--save_interval',metavar='[save interval in #epoch]',action='store',type=int,choices=range(0,10),default=1)
    parser.add_argument('--batch_size',metavar='[batch size]',action='store',type=int,choices=range(1,1000),default=64)
    parser.add_argument('--iter_per_patch',metavar='[how many iterations to run per batch]',action='store',type=int,choices=range(1,100),default=1)
    parser.add_argument('--keep_prob',metavar='[dropout rate]',action='store',type=float,default=0.9)
    parser.add_argument('--from_stepsize',metavar='[starting stepsize]',action='store',type=float,default=1e-4)
    parser.add_argument('--to_stepsize',metavar='[ending stepsize]',action='store',type=float,default=1e-6)
    parser.add_argument('--mean_file', dest='mean_file', action='store_true')
    parser.add_argument('--no_mean_file', dest='mean_file', action='store_false')
    parser.set_defaults(mean_file=True)
    return parser.parse_args()
#nn util
def weight_variable(shape, n='W', std=0.001):
    initial = tf.truncated_normal(shape,stddev=std)
    return tf.get_variable(initializer=initial,name=n)
def bias_variable(shape, n='b', std=0.001):
    initial = tf.truncated_normal(shape,stddev=std)
    return tf.get_variable(initializer=initial,name=n)
def conv2d(x, W, stride=1, pd='SAME'):
    return tf.nn.conv2d(x,W,strides=[1, stride, stride, 1],padding=pd)
def conv2d_transpose(x, W, stride=2, pd='SAME'):
    xshape=x.shape.as_list()
    Wshape=W.shape.as_list()
    batch_size=tf.shape(x)[0]
    oshape=[batch_size,xshape[1]*2,xshape[2]*2,Wshape[2]]
    return tf.nn.conv2d_transpose(value=x,filter=W,output_shape=oshape,strides=[1, stride, stride, 1],padding=pd)
def max_pool(x, ksize=2, stride=2, pd='SAME'):
    return tf.nn.max_pool(x,ksize=[1, ksize, ksize, 1],strides=[1, stride, stride, 1],padding=pd)
#network
def encoder(results, image, keep, nrk=[32,64,128, 128,128], resize=[128,128]):
    with tf.variable_scope('resize',reuse=tf.AUTO_REUSE):
        imresize=tf.image.resize_images(image,resize)
    with tf.variable_scope('conv1',reuse=tf.AUTO_REUSE):
        W=weight_variable([3,3,image.shape[3].value,nrk[0]])
        b=bias_variable([nrk[0]])
        h1=max_pool(conv2d(imresize,W)+b)
    with tf.variable_scope('conv2',reuse=tf.AUTO_REUSE):
        W=weight_variable([3,3,nrk[0],nrk[1]])
        b=bias_variable([nrk[1]])
        h2=max_pool(conv2d(h1,W)+b)
    with tf.variable_scope('conv3',reuse=tf.AUTO_REUSE):
        W=weight_variable([3,3,nrk[1],nrk[2]])
        b=bias_variable([nrk[2]])
        h3=max_pool(conv2d(h2,W)+b)
    with tf.variable_scope('fc1',reuse=tf.AUTO_REUSE):
        h3Shape=h3.shape.as_list()
        sprod=h3Shape[1]*h3Shape[2]*h3Shape[3]
        W=weight_variable([sprod,nrk[3]])
        b=bias_variable([nrk[3]])
        h3Flat=tf.reshape(h3,[-1,sprod])
        f1=tf.nn.relu(tf.matmul(h3Flat,W)+b)
    with tf.variable_scope('fc2',reuse=tf.AUTO_REUSE):
        W=weight_variable([nrk[3],nrk[4]])
        b=bias_variable([nrk[4]])
        f2=tf.nn.relu(tf.matmul(f1,W)+b)
    with tf.variable_scope('dropout',reuse=tf.AUTO_REUSE):
        df2=tf.nn.dropout(f2,keep)
    if results.debug_shape:
        print(h1.shape, h2.shape, h3.shape, h3Flat.shape, f1.shape, f2.shape)
    return imresize,df2,h3Shape,sprod
def decoder(results, image, feat, h3Shape, sprod, nrk=[32,64,128, 128,128], resize=[128,128]):
    with tf.variable_scope('fc2',reuse=tf.AUTO_REUSE):
        W=weight_variable([nrk[3],nrk[4]])
        b=bias_variable([nrk[4]])
        f2=tf.nn.relu(tf.matmul(feat,W)+b)
    with tf.variable_scope('fc1',reuse=tf.AUTO_REUSE):
        W=weight_variable([nrk[3],sprod])
        b=bias_variable([sprod])
        f1=tf.nn.relu(tf.matmul(f2,W)+b)
        h3=tf.reshape(f1,[-1,h3Shape[1],h3Shape[2],h3Shape[3]])
    with tf.variable_scope('conv3',reuse=tf.AUTO_REUSE):
        W=weight_variable([3,3,nrk[1],nrk[2]])
        b=bias_variable([nrk[1]])
        h2=tf.nn.relu(conv2d_transpose(h3,W)+b)
    with tf.variable_scope('conv2',reuse=tf.AUTO_REUSE):
        W=weight_variable([3,3,nrk[0],nrk[1]])
        b=bias_variable([nrk[0]])
        h1=tf.nn.relu(conv2d_transpose(h2,W)+b)
    with tf.variable_scope('conv1',reuse=tf.AUTO_REUSE):
        W=weight_variable([3,3,image.shape[3].value,nrk[0]])
        b=bias_variable([image.shape[3].value])
        imagePred=tf.image.resize_images(conv2d_transpose(h1,W)+b,resize)
    if results.debug_shape:
        print(f2.shape, h3.shape, h2.shape, h1.shape, imagePred.shape)
    return imagePred
def autoencoder_create(results,img):
    image=tf.placeholder(tf.float32,[None,img.shape[0],img.shape[1],img.shape[2]],name='image')
    keep=tf.placeholder(tf.float32,name='keepProb')
    with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
        imresize,feat,h3Shape,sprod=encoder(results,image,keep)
    with tf.variable_scope('decoder',reuse=tf.AUTO_REUSE):
        imagePred=decoder(results,image,feat,h3Shape,sprod)
    return image,imresize,feat,imagePred,keep
def autoencoder_create_trainer(results, image, imagePred):
    with tf.variable_scope('loss',reuse=tf.AUTO_REUSE):
        eucLoss=tf.nn.l2_loss(image-imagePred)
    trainStep,learning_rate,global_step=create_trainer(results,eucLoss,results.from_stepsize,results.to_stepsize)
    return trainStep,learning_rate,global_step,eucLoss
#parameters
def autoencoder_train(results, sess):
    ds=Dataset(results.dataset_path,results.mean_file)
    image,imresize,feat,imagePred,keep=autoencoder_create(results,ds.images[0])
    trainStep,learning_rate,global_step,eucLoss=autoencoder_create_trainer(results,imresize,imagePred)
    #saver
    saver=tf.train.Saver()
    if not os.path.exists(results.model_path):
        os.mkdir(results.model_path)
    #train
    sess.run(tf.global_variables_initializer())
    i=load_latest(results,sess)
    while i<results.max_epoch:
        #run one epoch
        ds.reset()
        while ds.nrEpoch() == 0:
            images=ds.nextBatch(results.batch_size,False)
            for i_per_batch in range(results.iter_per_patch):
                trainStep.run(feed_dict={keep:results.keep_prob,global_step:i,image:images})
        #test result
        rate=learning_rate.eval(feed_dict={global_step:i})
        loss=eucLoss.eval(feed_dict={keep:1,image:ds.images})
        print('Epoch %d, rate: %f, loss: %f'%(i,rate,loss))
        #write/profile
        if i%results.save_interval == 0:
            print('Saving %d epoch, batchSize=%d'%(i,results.batch_size))
            saver.save(sess,'%s/iter%d.ckpt'%(results.model_path,i))
            sys.stdout.flush()
        i=i+1

if __name__== "__main__":
    results=get_parser('autoencoder')
    #ds=Dataset(results.dataset_path,results.mean_file)
    #for i in range(1000):
    #    ds.nextBatch(64,random=True)
    #    ds.nextBatch(64,random=False)
    print('Using tensorflow: %s'%tf.__version__)
    results.debug_shape=True
    with tf.Session() as sess:
        autoencoder_train(results,sess)
