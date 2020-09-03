import os.path
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import helper
import warnings
from distutils.version import LooseVersion


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #set a tf graph for as container
    vgg_tag = 'vgg16'
    tf.saved_model.loader.load(sess,[vgg_tag],"/kaggle/working/vgg")
    graph=tf.Graph()
    graph=tf.get_default_graph()
    
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    vgg_keeprob_tensor = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    vgg_layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    vgg_layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    vgg_layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return input_image , vgg_keeprob_tensor , vgg_layer3_out , vgg_layer4_out , vgg_layer7_out

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    #take the last layer which is responsible to 32x upsample and use 1by1 conv with output kernels size=num classes
    conv_vgg7 = tf.layers.conv2d(vgg_layer7_out ,
                                num_classes ,
                                1 ,
                                strides=[1,1],
                                padding='same',
                                kernel_regularizer=tf.keras.regularizers.l2(1e-3))
    
    conv_2vgg7 = tf.layers.conv2d_transpose(conv_vgg7 , 
                                            filters=num_classes , 
                                            kernel_size=4 ,
                                            strides=[2,2],
                                            padding='same' ,
                                            kernel_regularizer=tf.keras.regularizers.l2(1e-3))
    #then the pool2 and conv_7_upsampled with same dimensions 
    conv_vgg4 = tf.layers.conv2d(vgg_layer4_out , 
                                filters=num_classes , 
                                kernel_size= 1 ,
                                strides=(1,1),
                                padding='same',
                                kernel_regularizer=tf.keras.regularizers.l2(1e-3))
    #add those tensors
    conv_add = tf.add(conv_vgg4 , conv_2vgg7)
    #use the upsample to match with output from vgg pool layer 3
    conv_2vgg4 = tf.layers.conv2d_transpose(conv_add , 
                                            filters=num_classes ,
                                            kernel_size= 4 ,
                                            strides= (2,2),
                                            padding='same',
                                            kernel_regularizer=tf.keras.regularizers.l2(1e-3))
    #convolute the vgg3 layer with 1by1 convolution with filter size = num_classes
    conv_vgg3 = tf.layers.conv2d(vgg_layer3_out , 
                                filters=num_classes,
                                kernel_size=1 ,
                                strides=(1,1),
                                padding='same',
                                kernel_regularizer=tf.keras.regularizers.l2(1e-3))
    #add the outputs 
    conv_add2 = tf.add(conv_vgg3 , conv_2vgg4)
    
    #totally upsample with stride 8 and kernerl size=16
    out_result = tf.layers.conv2d_transpose(conv_add2 , 
                                            filters=num_classes,
                                            kernel_size=16 ,
                                            strides=(8,8),
                                            padding='same',
                                            kernel_regularizer=tf.keras.regularizers.l2(1e-3))
    
    
    
    return out_result


def optimize(nn_last_layer, correct_label, learning_rate, num_classes,with_accuracy=True):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer , (-1,num_classes))
    labels = tf.reshape(correct_label,(-1,num_classes))
    #define the loss function
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits ,
                                                                labels=labels))
    #initialize the adam optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(cross_entropy_loss)
    if with_accuracy:
        predictions_argmax = tf.argmax(nn_last_layer, axis=-1)
        labels_argmax = tf.argmax(correct_label, axis=-1)
        iou,iou_op = tf.metrics.mean_iou(labels_argmax , predictions_argmax , num_classes)
        iou_obj =[iou , iou_op]
        
        return logits , train_op , cross_entropy_loss , iou_obj
    else:
        return logits , train_op , cross_entropy_loss , None
    
def restore_model(sess, model_path):
    """
    Restore session, which was previously saved to file.
    :param sess: TF Session
    :param model_path: path to several files having model_path as their prefix
    """
    saver = tf.train.import_meta_graph(model_path + '.meta')
    saver.restore(sess, model_path)