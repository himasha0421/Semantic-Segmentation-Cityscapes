import re
import random
import numpy as np
import os.path
import scipy.misc
import shutil
import zipfile
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from glob import glob
from urllib.request import urlretrieve
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np 
import torch.nn.functional as F
import os.path
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import helper
import warnings
from distutils.version import LooseVersion


process_load = True
def run():
    epochs=60
    num_classes = 19
    batch_size = 32
    image_shape = (224, 224)  # KITTI dataset uses 160x576 images
    data_dir = '/kaggle/working'
    runs_dir = '/runs'

    # Download pretrained vgg model
    maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    correct_label = tf.placeholder(tf.float32 , [None , None ,None , num_classes])
    learning_rate = tf.placeholder(tf.float32)

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = gen_batch_function(image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        input_image , keep_prob , vgg_layer3 , vgg_layer4 , vgg_layer7 = load_vgg(sess,vgg_path)
        
        last_layer = layers(vgg_layer3 ,  vgg_layer4 , vgg_layer7 , num_classes)
        
        logits , train_op , cross_entropy_loss , iou_obj = optimize(last_layer , correct_label ,learning_rate ,num_classes )
        
        #initialize all the gloabl and local defined variables for the training
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver()
        if(process_load):
            checkpoint = tf.train.get_checkpoint_state("/content/gdrive/My Drive/cityscape_model")
            if checkpoint and checkpoint.model_checkpoint_path:
                saver.restore(sess, checkpoint.model_checkpoint_path)
                print("Successfully loaded:", checkpoint.model_checkpoint_path)
            else:
                print("Could not find old network weights")
        
        #run the training for the images
        train_nn(sess , epochs ,batch_size ,get_batches_fn ,train_op , cross_entropy_loss ,input_image ,correct_label ,keep_prob ,learning_rate , iou_obj ,logits )


        # TODO: Save inference data using helper.save_inference_samples
        save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        
def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate ,acc_obj ,logits  ):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    #run the trainer on all the epochs
    for epoch in range(epochs) :
        epoch_loss = 0
        image_cout = 0
        total_acc = 0
        batch_count=0
        #get the batches to train
        for batch_x , batch_y in get_batches_fn(batch_size):
            #in the training optimization use the loss and trainop to exucution
            _,loss = sess.run([train_op , cross_entropy_loss] ,feed_dict={
                input_image:batch_x ,
                correct_label:batch_y ,
                keep_prob:0.5 ,
                learning_rate:0.0001 
            })
            if(acc_obj is not None):
                iou = acc_obj[0]
                iou_op = acc_obj[1]
                sess.run(iou_op,feed_dict={
                    input_image:batch_x ,
                    correct_label:batch_y ,
                    keep_prob:1.0 ,
                    learning_rate:0.001
                })
                
                acc = sess.run(iou)
                total_acc += acc * len(batch_x)
            image_cout+=len(batch_x)
            batch_count +=1
            epoch_loss  += loss 
        epoch_acc = total_acc / image_cout
        epoch_loss = epoch_loss / batch_count
        #visualize the resulting logits
        #take a random image and show the image from last batch
        rand_idx = np.random.randint(0,len(batch_x))
        random_img,random_label= batch_x[rand_idx],batch_y[rand_idx]
        rand_img=random_img.reshape(1,*random_img.shape)
        rand_label =random_label.reshape(1,*random_label.shape)
        
        result ,loss = sess.run([logits , cross_entropy_loss] , feed_dict={
            input_image:rand_img ,
            correct_label:rand_label ,
            keep_prob:1.0 ,
            learning_rate:0.001
        })
        if(epoch%5==0):
            plot_results(random_img , np.squeeze(result))
        if((epoch+1)%10==0):
            saver = tf.train.Saver()
            saver.save(sess, "/content/gdrive/My Drive/cityscape_model/model.ckpt")
        print("Training >>>>> Epoch : {} Epoch Loss : {} Epoch accuracy : {:.3f}".format(epoch ,epoch_loss , epoch_acc))

    
    pass