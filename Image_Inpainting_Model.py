import numpy as np
import tensorflow as tf
import cv2
#batch size is constant:16(conv2d_transpose can't accept variable batch size)
#encoder decoder architecture
#encoder: 4 layer convnet trained from scratch followed by channel-wise fc layer followed by a 2d-convolution
#decoder: 5 upconvolution layers reconstructing the full image
#encoder-decoder is trained end-to-end
def build_model(X,reuse):
    batch_size=X.get_shape().as_list()[0]
    with tf.variable_scope("gen",reuse=reuse) as genScope:
        with tf.variable_scope('encoder') as outScope:
            with tf.variable_scope('conv1') as scope:
                W=tf.get_variable(initializer=tf.random_normal_initializer(mean=0.0,stddev=0.001),shape=(4,4,3,64),name='filter')
                B=tf.get_variable(name='bias',initializer=tf.constant(0.1,shape=(64,)))
                conv1=tf.nn.relu(tf.nn.conv2d(X,W,strides=[1,2,2,1],padding='SAME') + B)
            print conv1.get_shape()
            with tf.variable_scope('conv2') as scope:
                W=tf.get_variable(initializer=tf.random_normal_initializer(mean=0.0,stddev=0.001),shape=(4,4,64,64),name='filter')
                B=tf.get_variable(name='bias',initializer=tf.constant(0.1,shape=(64,)))
                conv2=tf.nn.relu(tf.nn.conv2d(conv1,W,strides=[1,2,2,1],padding='SAME') + B)
            print conv2.get_shape()
            with tf.variable_scope('conv3') as scope:
                W=tf.get_variable(initializer=tf.random_normal_initializer(mean=0.0,stddev=0.001),shape=(4,4,64,128),name='filter')
                B=tf.get_variable(name='bias',initializer=tf.constant(0.1,shape=(128,)))
                conv3=tf.nn.relu(tf.nn.conv2d(conv2,W,strides=[1,2,2,1],padding='SAME') + B)
            print conv3.get_shape()
            with tf.variable_scope('conv4') as scope:
                W=tf.get_variable(initializer=tf.random_normal_initializer(mean=0.0,stddev=0.001),shape=(4,4,128,256),name='filter')
                B=tf.get_variable(name='bias',initializer=tf.constant(0.1,shape=(256,)))
                conv4=tf.nn.relu(tf.nn.conv2d(conv3,W,strides=[1,2,2,1],padding='VALID') + B)
            print conv4.get_shape()
        batch_size,h,w,nFeat=conv4.get_shape().as_list()
        out_to_cfc=tf.reshape(conv4,[-1,h*w,nFeat])
        W=tf.get_variable(name='ChannelWiseFC',initializer=tf.random_normal_initializer(mean=0.0,stddev=0.001),shape=(nFeat,h*w,h*w))
        out_from_enc=tf.transpose(out_to_cfc,(2,0,1))
        inp_to_dec=tf.batch_matmul(out_from_enc,W)
        decoder_input=tf.reshape(inp_to_dec,(-1,h,w,nFeat))
        print decoder_input.get_shape()

        with tf.variable_scope('DecodeConv') as scope:
            W=tf.get_variable(initializer=tf.random_normal_initializer(mean=0.0,stddev=0.001),shape=(4,4,256,512),name='filter')
            B=tf.get_variable(name='bias',initializer=tf.constant(0.1,shape=(512,)))
            convOut=tf.nn.relu(tf.nn.conv2d(decoder_input,W,strides=[1,1,1,1],padding='SAME')+B)

        with tf.variable_scope('decoder') as outScope:  
            with tf.variable_scope('deconv4') as scope:
                W=tf.get_variable(initializer=tf.random_normal_initializer(mean=0.0,stddev=0.001),shape=(4,4,256,512),name='filter')
                B=tf.get_variable(name='bias',initializer=tf.constant(0.1,shape=(256,)))
                deconv4=tf.nn.relu(tf.nn.conv2d_transpose(convOut,W,decoder_input.get_shape(),[1,1,1,1],padding='SAME') + B)
                
            with tf.variable_scope('deconv3') as scope:
                W=tf.get_variable(initializer=tf.random_normal_initializer(mean=0.0,stddev=0.001),shape=(4,4,128,256),name='filter')
                B=tf.get_variable(name='bias',initializer=tf.constant(0.1,shape=(128,)))
                deconv3=tf.nn.relu(tf.nn.conv2d_transpose(deconv4,W,conv3.get_shape(),[1,2,2,1],padding='VALID')+B)

            print deconv3.get_shape()

            with tf.variable_scope('deconv2') as scope:
                W=tf.get_variable(initializer=tf.random_normal_initializer(mean=0.0,stddev=0.001),shape=(4,4,64,128),name='filter')
                b=tf.get_variable(name='bias',initializer=tf.constant(0.1,shape=(64,)))
                deconv2=tf.nn.relu(tf.nn.conv2d_transpose(deconv3,W,conv2.get_shape(),[1,2,2,1],padding='SAME')+b)

            print deconv2.get_shape()
            with tf.variable_scope('deconv1') as scope:
                W=tf.get_variable(initializer=tf.random_normal_initializer(mean=0.0,stddev=0.001),shape=(4,4,3,64),name='filter')
                b=tf.get_variable(name='bias',initializer=tf.constant(0.1,shape=(3,)))
                deconv1=tf.nn.relu(tf.nn.conv2d_transpose(deconv2,W,[batch_size,64,64,3],[1,2,2,1],padding='SAME')+b)
            print deconv1.get_shape()
        print "Model built"
    return conv1,conv2,conv3,conv4,decoder_input,convOut,deconv4,deconv3,deconv2,deconv1

def build_adversarial(X,reuse):
    print "Building adversarial discriminator"
    with tf.variable_scope('Dis',reuse=reuse):
        with tf.variable_scope('conv1'):
            W=tf.get_variable(initializer=tf.random_normal_initializer(mean=0.0,stddev=0.001),shape=(4,4,3,64),name='filter')
            B=tf.get_variable(name='bias',initializer=tf.constant(0.1,shape=(64,)))
            conv1=tf.nn.relu(tf.nn.conv2d(X,W,strides=[1,2,2,1],padding='SAME') + B)

        with tf.variable_scope('conv2') as scope:
            W=tf.get_variable(initializer=tf.random_normal_initializer(mean=0.0,stddev=0.001),shape=(4,4,64,64),name='filter')
            B=tf.get_variable(name='bias',initializer=tf.constant(0.1,shape=(64,)))
            conv2=tf.nn.relu(tf.nn.conv2d(conv1,W,strides=[1,2,2,1],padding='SAME') + B)
        print conv2.get_shape()
        with tf.variable_scope('conv3') as scope:
            W=tf.get_variable(initializer=tf.random_normal_initializer(mean=0.0,stddev=0.001),shape=(4,4,64,128),name='filter')
            B=tf.get_variable(name='bias',initializer=tf.constant(0.1,shape=(128,)))
            conv3=tf.nn.relu(tf.nn.conv2d(conv2,W,strides=[1,2,2,1],padding='SAME') + B)
        print conv3.get_shape()
        with tf.variable_scope('conv4') as scope:
            W=tf.get_variable(initializer=tf.random_normal_initializer(mean=0.0,stddev=0.001),shape=(4,4,128,256),name='filter')
            B=tf.get_variable(name='bias',initializer=tf.constant(0.1,shape=(256,)))
            conv4=tf.nn.relu(tf.nn.conv2d(conv3,W,strides=[1,2,2,1],padding='VALID') + B)
        print conv4.get_shape()
        with tf.variable_scope('Disc_End') as scope:
            input_shape=conv4.get_shape().as_list()
            input_dim=np.prod(input_shape[1:])
            output_dim=1
            x=tf.reshape(conv4,shape=[-1,input_dim])
            print x.get_shape()
            W=tf.get_variable(initializer=tf.random_normal_initializer(mean=0.0,stddev=0.001),shape=(input_dim,output_dim),name='weight')
            b=tf.get_variable(name='bias',initializer=tf.constant_initializer(0.1),shape=[output_dim])
            print W.get_shape()
            output=tf.nn.bias_add(tf.matmul(x,W),b)
        print output.get_shape()
        return output