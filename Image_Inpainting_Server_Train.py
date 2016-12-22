import numpy as np
import tensorflow as tf
import cv2
import os
import glob
from Image_Inpainting_Model import build_model
from Image_Inpainting_Model import build_adversarial
mean=113.85303206371599
stddev=61.135735318812173
lambda_recon=0.9
lambda_adv=0.1
reuse=None
learning_rate_val=0.0001
tf.reset_default_graph()
def read_batch(path_to_files,mean=mean,stddev=stddev):
    imgs_inp=[]
    imgs_lab=[]
    os.chdir('/root/sharedfolder/Image_Inpainting/Dataset/')
    for file in path_to_files:
        inp=cv2.imread('LFW_Input/' + file.split('/')[-1])
        lab=cv2.imread('LFW_Label/' + file.split('/')[-1])
        imgs_inp.append((inp-mean)/stddev)
        imgs_lab.append((lab-mean)/stddev)
    return np.array(imgs_inp),np.array(imgs_lab)
sess = tf.InteractiveSession()
X = tf.placeholder(tf.float32,shape=[16,128,128,3])
y = tf.placeholder(tf.float32,shape=[16,64,64,3])
imgs_hiding = tf.placeholder(tf.float32,shape=[16,64,64,3]) #central pixels
#conv1,conv2,conv3,conv4,convOut,deconv4,deconv3,deconv2,deconv1 = build_model(X,reuse) #deconv1 is the reconstruction(64,64,3)
conv1,conv2,conv3,conv4,decoder_input,convOut,deconv4,deconv3,deconv2,deconv1=build_model(X,reuse)
batch_size = X.get_shape().as_list()[0]
labels_d = tf.reshape(tf.concat(0,(tf.ones(batch_size),tf.zeros(batch_size))),[2*batch_size,1])
labels_g = tf.reshape(tf.ones(batch_size),[batch_size,1])
adv_pos = build_adversarial(imgs_hiding,reuse)
reuse = True
adv_neg = build_adversarial(deconv1,reuse) #running generated images through discriminator
adv_all = tf.concat(0,[adv_pos,adv_neg])
print "GAN output shape " + str(adv_all.get_shape().as_list())
loss_recon = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(deconv1,y)))) #no Mcap, already predicting the central pixels
loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(adv_neg,labels_g))
loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(adv_all,labels_d))
loss_gen = lambda_adv*loss_g + lambda_recon*loss_recon
loss_disc = lambda_adv*loss_d 
optim_d = tf.train.AdamOptimizer(learning_rate=learning_rate_val).minimize(loss_disc)
optim_g = tf.train.AdamOptimizer(learning_rate=learning_rate_val).minimize(loss_gen)

Mcap=np.zeros((128,128,3))
Mcap[32:96,32:96,:]=1
for v in tf.trainable_variables():
    print v.name
#train_step=tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
tf.initialize_all_variables().run()
files=glob.glob('/root/sharedfolder/Image_Inpainting/Dataset/LFW_Label/*.jpg')
nImages=int(raw_input("Enter number of images "))
nIter=int(raw_input("Enter number of iterations "))
learning_rate_val=float(raw_input("Enter learning rate"))
n=nImages/16
print str(n*nIter) + ' iterations'
f=open('Curr_Batch_' + str(nImages) + '_' + str(nIter) + '.txt','a')
for i in range(n*nIter):
    curr_batch=np.random.choice(files,16)
    np.savetxt(f,curr_batch,fmt='%s',delimiter=',')
    imgs_inp,imgs_lab=read_batch(curr_batch)
    sess.run(optim_g,feed_dict={X:imgs_inp,y:imgs_lab})
    if(i%10==0):
        sess.run(optim_d,feed_dict={X:imgs_inp,y:imgs_lab,imgs_hiding:imgs_lab})
    if(i%100==0):
        loss_l2,loss_gener,loss_discrim=sess.run([loss_recon,loss_gen,loss_disc],feed_dict={X:imgs_inp,y:imgs_lab,imgs_hiding:imgs_lab})
        print("Epoch:%d Reconstruction loss: %g Generator loss: %g Discriminator loss: %g"%(i+1,loss_l2,loss_gener,loss_discrim))
    if(i%500==0):  #generating images giving current batch
        conv1,conv2,conv3,conv4,decoder_input,convOut,deconv4,deconv3,deconv2,deconv1=build_model(X,True)
        imgs=sess.run(deconv1,feed_dict={X:imgs_inp})
        curr,_=read_batch(curr_batch,mean=0,stddev=1)
        for ind in range(len(imgs)):
            currImg=curr[ind]
            pred=(imgs[ind]*stddev)+mean
            print np.mean(pred),np.std(pred)
            currImg[32:96,32:96,:]=pred
            cv2.imwrite('IMG_'+ str(i) + '_' + str(ind) + '.jpg',currImg)
f.close()
saver=tf.train.Saver()
saver.save(sess,'Model_' + str(nImages) +'_' +str(nIter) + '.ckpt')
