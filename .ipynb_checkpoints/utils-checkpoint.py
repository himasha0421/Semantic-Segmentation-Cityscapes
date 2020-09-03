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

class DLProgress(tqdm):
	"""
	Report download progress to the terminal.
	:param tqdm: Information fed to the tqdm library to estimate progress.
	"""
	last_block = 0

	def hook(self, block_num=1, block_size=1, total_size=None):
		"""
		Store necessary information for tracking progress.
		:param block_num: current block of the download
		:param block_size: size of current block
		:param total_size: total download size, if known
		"""
		self.total = total_size
		self.update((block_num - self.last_block) * block_size)  # Updates progress
		self.last_block = block_num


def maybe_download_pretrained_vgg(data_dir):
	"""
	Download and extract pretrained vgg model if it doesn't exist
	:param data_dir: Directory to download the model to
	"""
	vgg_filename = 'vgg.zip'
	vgg_path = os.path.join(data_dir, 'vgg')
	vgg_files = [
		os.path.join(vgg_path, 'variables/variables.data-00000-of-00001'),
		os.path.join(vgg_path, 'variables/variables.index'),
		os.path.join(vgg_path, 'saved_model.pb')]

	missing_vgg_files = [vgg_file for vgg_file in vgg_files if not os.path.exists(vgg_file)]
	if missing_vgg_files:
		# Clean vgg dir
		if os.path.exists(vgg_path):
			shutil.rmtree(vgg_path)
		os.makedirs(vgg_path)

		# Download vgg
		print('Downloading pre-trained vgg model...')
		with DLProgress(unit='B', unit_scale=True, miniters=1) as pbar:
			urlretrieve(
				'https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',
				os.path.join(vgg_path, vgg_filename),
				pbar.hook)

		# Extract vgg
		print('Extracting model...')
		zip_ref = zipfile.ZipFile(os.path.join(vgg_path, vgg_filename), 'r')
		zip_ref.extractall(data_dir)
		zip_ref.close()

		# Remove zip file to save space
		os.remove(os.path.join(vgg_path, vgg_filename))


def gen_batch_function( image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        
        print("Working")
        image_paths=[]
        label_paths=[]
        counter=0
        for dirname, _, filenames in os.walk('/content/cityscape_train/'):
            counter+=1
            
            for filename in filenames:
                #img_path = os.path.join(dirname, filename)
                #image_paths.append(img_path)
       
                for dirname1, _, filenames1 in os.walk('/content/get_index/train'):
                    for filename1 in filenames1:
                        if(filename1==filename):
                           
                            gd_path = os.path.join(dirname1, filename1)
                            label_paths.append(gd_path)
                            img_path = os.path.join(dirname, filename)
                            image_paths.append(img_path)
        # Loop through batches and grab images, yielding each batch
        for batch_i in range(0, len(image_paths), batch_size):
            images = []
            gt_images = []
            for idx , image_file in enumerate(image_paths[batch_i:batch_i+batch_size]):
                gt_image_file = label_paths[batch_i+idx]
                
                # Re-size to image_shape
                image = cv2.imread(image_file)
                image = cv2.resize(image, (224,224), interpolation = cv2.INTER_AREA)
                gt_image = cv2.imread(gt_image_file,cv2.IMREAD_GRAYSCALE)
                gt_image = cv2.resize(gt_image, (224,224), interpolation = cv2.INTER_AREA)
                # Create "one-hot-like" labels by class
                color_list=[]
                for color_id in range(19):
                    gt_bg = np.array(gt_image == color_id).astype(np.uint8)
                    gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                    color_list.append(gt_bg)
                gt_image = np.concatenate(color_list, axis=2)
                images.append(image)
                gt_images.append(gt_image)

            yield np.array(images), np.array(gt_images)
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
	"""
	Generate test output using the test images
	:param sess: TF session
	:param logits: TF Tensor for the logits
	:param keep_prob: TF Placeholder for the dropout keep probability
	:param image_pl: TF Placeholder for the image placeholder
	:param data_folder: Path to the folder that contains the datasets
	:param image_shape: Tuple - Shape of image
	:return: Output for for each test image
	"""
	for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
		image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

		# Run inference
		im_softmax = sess.run(
			[tf.nn.softmax(logits)],
			{keep_prob: 1.0, image_pl: [image]})
		# Splice out second column (road), reshape output back to image_shape
		im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
		# If road softmax > 0.5, prediction is road
		segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
		# Create mask based on segmentation to apply to original image
		mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
		mask = scipy.misc.toimage(mask, mode="RGBA")
		street_im = scipy.misc.toimage(image)
		street_im.paste(mask, box=None, mask=mask)

		yield os.path.basename(image_file), np.array(street_im)


def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image):
	"""
	Save test images with semantic masks of lane predictions to runs_dir.
	:param runs_dir: Directory to save output images
	:param data_dir: Path to the directory that contains the datasets
	:param sess: TF session
	:param image_shape: Tuple - Shape of image
	:param logits: TF Tensor for the logits
	:param keep_prob: TF Placeholder for the dropout keep probability
	:param input_image: TF Placeholder for the image placeholder
	"""
	# Make folder for current run
	output_dir = os.path.join(runs_dir, str(time.time()))
	if os.path.exists(output_dir):
		shutil.rmtree(output_dir)
	os.makedirs(output_dir)

	# Run NN on test images and save them to HD
	print('Training Finished. Saving test images to: {}'.format(output_dir))
	image_outputs = gen_test_output(
		sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
	for name, image in image_outputs:
		scipy.misc.imsave(os.path.join(output_dir, name), image)
        

# PyTroch version

SMOOTH = 1e-6
fig = plt.figure(figsize=(10,10))
def plot_results(ori_image , result_img):
        plot_img(ori_image)
        class_image = iou_result(result_img)
        convert(class_image)


def iou_result(outputs):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.reshape(224,224,19)
    outputs = np.transpose(outputs , (2,0,1))
    outputs = torch.tensor(outputs , dtype=torch.float , device='cpu').unsqueeze(0)
    output_prob = F.log_softmax(outputs , dim=1)
    #take the argmax value in the dim 1
    output_class = torch.argmax(output_prob , dim=1)
    output_class = output_class.squeeze(0)
    return output_class # Or thresholded.mean() if you are interested in average across the batch


def convert(mask):
    mask = mask.numpy()
    height , width = mask.shape
    copy_img = np.zeros((height , width , 3) , dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            idx = mask[i][j]
            label = labels[idx]
            color = label.color
            copy_img[i][j]=color
    plt.title("Segment Mask")
    plt.imshow(copy_img)
    plt.show()

def plot_img(image):
    #unnormallize the image
    plt.title("Original Image")
    plt.imshow(image)
    plt.show()