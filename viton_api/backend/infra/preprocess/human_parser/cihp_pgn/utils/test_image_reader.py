import os

import numpy as np
import tensorflow as tf
import random

IGNORE_LABEL = 255
IMG_MEAN = np.array((125.0, 114.4, 107.9), dtype=np.float32)

def read_images_from_disk(input_queue):  # optional pre-processing arguments
    """Read one image and its corresponding mask with optional pre-processing.
    
    Args:
      input_queue: tf queue with paths to the image and its mask.
      
    Returns:
      Two tensors: the decoded image and its mask.
    """

    img_contents = tf.read_file(input_queue[0])

    img = tf.image.decode_jpeg(img_contents, channels=3)
    img_r, img_g, img_b = tf.split(value=img, num_or_size_splits=3, axis=2)
    img = tf.cast(tf.concat([img_b, img_g, img_r], 2), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN

    return img

def read_image_list(data_dir):
    """Reads txt file containing paths to images and ground truth masks.
    
    Args:
      data_dir: path to the directory with images and masks.
       
    Returns:
      Two lists with all file names for images and masks, respectively.
    """
    
    images = []
    for img_name in os.listdir(data_dir):
        images.append(data_dir + '/' + img_name)
    return images

class TestImageReader(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_dir, data_list, data_id_list, input_size,
                 random_scale,
                 random_mirror, shuffle, coord):
        '''Initialise an ImageReader.
        
        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
          data_id_list: path to the file of image id.
          input_size: a tuple with (height, width) values, to which all the images will be resized.
          random_scale: whether to randomly scale the images prior to random crop.
          random_mirror: whether to randomly mirror the images prior to random crop.
          coord: TensorFlow queue coordinator.
        '''
        self.data_dir = data_dir
        self.data_list = data_list
        self.data_id_list = data_id_list
        self.input_size = input_size
        self.coord = coord

        self.image_list = read_image_list(self.data_dir)

        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.queue = tf.compat.v1.train.slice_input_producer([self.images],
                                                             shuffle=shuffle)
        print(self.queue)
        self.image = read_images_from_disk(self.queue)

    def dequeue(self, num_elements):
        '''Pack images and labels into a batch.
        
        Args:
          num_elements: the batch size.
          
        Returns:
          Two tensors of size (batch_size, h, w, {3, 1}) for images and masks.'''
        batch_list = [self.image, self.label, self.edge]
        image_batch, label_batch, edge_batch = tf.train.batch(
            [self.image, self.label, self.edge], num_elements)
        return image_batch, label_batch, edge_batch


class TestImageReaderCustom(object):
    '''Generic ImageReader which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_path, data_list, data_id_list, input_size,
                 random_scale, random_mirror, shuffle, coord):
        '''Initialise an ImageReader.
        
        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
          data_id_list: path to the file of image id.
          input_size: a tuple with (height, width) values, to which all the images will be resized.
          random_scale: whether to randomly scale the images prior to random crop.
          random_mirror: whether to randomly mirror the images prior to random crop.
          coord: TensorFlow queue coordinator.
        '''
        self.data_path = data_path
        self.data_list = data_list
        self.data_id_list = data_id_list
        self.input_size = input_size
        self.coord = coord

        self.image_list = [self.data_path]

        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.queue = tf.compat.v1.train.slice_input_producer([self.images],
                                                             shuffle=shuffle)
        print(self.queue)
        self.image = read_images_from_disk(self.queue)

    def dequeue(self, num_elements):
        '''Pack images and labels into a batch.
        
        Args:
          num_elements: the batch size.
          
        Returns:
          Two tensors of size (batch_size, h, w, {3, 1}) for images and masks.'''
        batch_list = [self.image, self.label, self.edge]
        image_batch, label_batch, edge_batch = tf.train.batch(
            [self.image, self.label, self.edge], num_elements)
        return image_batch, label_batch, edge_batch
