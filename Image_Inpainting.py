import numpy as np
import tensorflow as tf
import cv2
#batch size is constant:16(conv2d_transpose can't accept variable batch size)
#encoder decoder architecture
#encoder: 4 layer convnet trained from scratch followed by channel-wise fc layer followed by a 2d-convolution
#decoder: 5 upconvolution layers reconstructing the full image

def read_batch(n,files):
    Mcap=np.zeros((32,32,3))
    Mcap[8:-8,8:-8,:]=1
    imgs_inp=[]
    imgs_lab=[]
    for i in range(n):
        img=cv2.imread(files[i])
        img_inp=(1-Mcap)*img
        imgs_inp.append(img_inp)
        imgs_lab.append(img)
    return np.array(imgs_inp),np.array(imgs_lab)

def build_model():
    X=tf.placeholder(tf.float32,shape=(16,32,32,3))
    y=tf.placeholder(tf.float32,shape=(16,32,32,3))
    with tf.name_scope('encoder') as outScope:
        with tf.name_scope('conv1') as scope:
            W=tf.Variable(tf.truncated_normal(stddev=0.01,mean=0.0,shape=(4,4,3,64)),name='filter')
            B=tf.Variable(tf.constant(0.1,shape=(64,)),name='bias')
            conv1=tf.nn.relu(tf.nn.conv2d(X,W,strides=[1,1,1,1],padding='VALID') + B)
        print conv1.get_shape()
        with tf.name_scope('conv2') as scope:
            W=tf.Variable(tf.truncated_normal(stddev=0.01,mean=0.0,shape=(4,4,64,64)),name='filter')
            B=tf.Variable(tf.constant(0.1,shape=(64,)),name='bias')
            conv2=tf.nn.relu(tf.nn.conv2d(conv1,W,strides=[1,1,1,1],padding='VALID') + B)
        print conv2.get_shape()
        with tf.name_scope('conv3') as scope:
            W=tf.Variable(tf.truncated_normal(stddev=0.01,shape=(4,4,64,128)),name='filter')
            B=tf.Variable(tf.constant(0.1,shape=(128,)),name='bias')
            conv3=tf.nn.relu(tf.nn.conv2d(conv2,W,strides=[1,1,1,1],padding='VALID') + B)
        print conv3.get_shape()
        with tf.name_scope('conv4') as scope:
            W=tf.Variable(tf.truncated_normal(stddev=0.01,mean=0.0,shape=(4,4,128,256)),name='filter')
            B=tf.Variable(tf.constant(0.1,shape=(256,)),name='bias')
            conv=tf.nn.conv2d(conv3,W,strides=[1,1,1,1],padding='VALID')
            conv4=tf.nn.relu(conv + B)
        print conv4.get_shape()
    batch_size,h,w,nFeat=conv4.get_shape().as_list()
    out_to_cfc=tf.reshape(conv4,[-1,h*w,nFeat])
    W=tf.Variable(tf.truncated_normal(mean=0.0,stddev=0.001,shape=(nFeat,h*w,h*w)),name='Enc2Dec')
    out_from_enc=tf.transpose(out_to_cfc,(2,0,1))
    inp_to_dec=tf.batch_matmul(out_from_enc,W)
    decoder_input=tf.reshape(inp_to_dec,(-1,h,w,nFeat))
    print decoder_input.get_shape()

    with tf.name_scope('Enc2Dec') as scope:
        W=tf.Variable(tf.truncated_normal(stddev=0.01,mean=0.0,shape=(4,4,256,512)),name='filter')
        B=tf.Variable(tf.constant(0.1,shape=(512,)),name='bias')
        convOut=tf.nn.relu(tf.nn.conv2d(decoder_input,W,strides=[1,1,1,1],padding='SAME')+B)

    with tf.name_scope('decoder') as outScope:  
        with tf.name_scope('deconv5') as scope:
            W=tf.Variable(tf.truncated_normal(stddev=0.01,mean=0.0,shape=(4,4,256,512)),name='filter')
            B=tf.Variable(tf.constant(0.1,shape=(256,)),name='bias')
            deconv5=tf.nn.relu(tf.nn.conv2d_transpose(convOut,W,decoder_input.get_shape(),[1,1,1,1],padding='SAME') + B)
            
        with tf.name_scope('deconv4') as scope:
            W=tf.Variable(tf.truncated_normal(stddev=0.01,mean=0.0,shape=(4,4,128,256)),name='filter')
            B=tf.Variable(tf.constant(0.1,shape=(128,)),name='bias')
            deconv4=tf.nn.relu(tf.nn.conv2d_transpose(deconv5,W,conv3.get_shape(),[1,1,1,1],padding='VALID')+B)

        print deconv4.get_shape()

        with tf.name_scope('deconv3') as scope:
            W=tf.Variable(tf.truncated_normal(stddev=0.01,mean=0.0,shape=(4,4,64,128)),name='filter')
            b=tf.Variable(tf.constant(0.1,shape=(64,)),name='bias')
            deconv3=tf.nn.relu(tf.nn.conv2d_transpose(conv3,W,conv2.get_shape(),[1,1,1,1],padding='VALID')+b)

        print deconv3.get_shape()
        with tf.name_scope('deconv2') as scope:
            W=tf.Variable(tf.truncated_normal(stddev=0.01,mean=0.0,shape=(4,4,64,64)),name='filter')
            b=tf.Variable(tf.constant(0.1,shape=(64,)))
            deconv2=tf.nn.relu(tf.nn.conv2d_transpose(deconv3,W,conv1.get_shape(),[1,1,1,1],padding='VALID')+b)
        print deconv2.get_shape()

        with tf.name_scope('deconv1') as scope:
            W=tf.Variable(tf.truncated_normal(stddev=0.01,mean=0.0,shape=(4,4,3,64)),name='filter')
            b=tf.Variable(tf.constant(0.1,shape=(3,)))
            deconv1=tf.nn.relu(tf.nn.conv2d_transpose(deconv2,W,X.get_shape(),[1,1,1,1],padding='VALID')+b)
        print deconv1.get_shape()
        print deconv1.get_shape(),y.get_shape()
        print "Model built"
    Mcap=np.zeros((32,32,3))
    Mcap[8:-8,8:-8,:]=1
    loss=tf.reduce_mean(Mcap*tf.square(tf.sub(deconv1,y)))
    train_step=tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

def train(nIter):
    for i in range(nIter):

