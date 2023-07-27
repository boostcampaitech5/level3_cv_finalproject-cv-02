from __future__ import print_function

import tensorflow as tf

import os
import os.path as osp

import cv2
import numpy as np
from PIL import Image

from infra.preprocess.human_parser.utils import *


class CIHP_PGN:
    def __init__(self):
        self.n_classes = 20
        self.restore_from = "./checkpoint/CIHP_pgn"
        self.num_steps = 1
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.net_100 = None
        self.net_050 = None
        self.net_075 = None
        self.net_125 = None
        self.net_150 = None
        self.net_175 = None
        self.create_model()
        
        # Set up tf session and initialize variables. 
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config)
        self.init = tf.global_variables_initializer()
        
        self.sess.run(self.init)
        self.sess.run(tf.local_variables_initializer())
        
        # Load weights.
        self.loader = tf.compat.v1.train.Saver(var_list=tf.global_variables())
        if self.restore_from is not None:
            if self.load(self.loader, self.sess, self.restore_from):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
                return
    
    def create_model(self):
        """Create the PGN model."""
        # Define the placeholder for input image
        self.image_placeholder = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 3])
        
        h_orig, w_orig = tf.to_float(tf.shape(self.image_placeholder)[1]), tf.to_float(tf.shape(self.image_placeholder)[2])
        image_batch050 = tf.image.resize_images(self.image_placeholder, tf.stack([tf.to_int32(tf.multiply(h_orig, 0.50)), tf.to_int32(tf.multiply(w_orig, 0.50))]))
        image_batch075 = tf.image.resize_images(self.image_placeholder, tf.stack([tf.to_int32(tf.multiply(h_orig, 0.75)), tf.to_int32(tf.multiply(w_orig, 0.75))]))
        image_batch125 = tf.image.resize_images(self.image_placeholder, tf.stack([tf.to_int32(tf.multiply(h_orig, 1.25)), tf.to_int32(tf.multiply(w_orig, 1.25))]))
        image_batch150 = tf.image.resize_images(self.image_placeholder, tf.stack([tf.to_int32(tf.multiply(h_orig, 1.50)), tf.to_int32(tf.multiply(w_orig, 1.50))]))
        image_batch175 = tf.image.resize_images(self.image_placeholder, tf.stack([tf.to_int32(tf.multiply(h_orig, 1.75)), tf.to_int32(tf.multiply(w_orig, 1.75))]))        
        
        # Create network.
        with tf.compat.v1.variable_scope('', reuse=tf.compat.v1.AUTO_REUSE):
            self.net_100 = PGNModel({'data': self.image_placeholder}, is_training=False, n_classes=self.n_classes)
        
        with tf.compat.v1.variable_scope('', reuse=tf.compat.v1.AUTO_REUSE):
            self.net_050 = PGNModel({'data': image_batch050}, is_training=False, n_classes=self.n_classes)
        
        with tf.compat.v1.variable_scope('', reuse=tf.compat.v1.AUTO_REUSE):
            self.net_075 = PGNModel({'data': image_batch075}, is_training=False, n_classes=self.n_classes)
        
        with tf.compat.v1.variable_scope('', reuse=tf.compat.v1.AUTO_REUSE):
            self.net_125 = PGNModel({'data': image_batch125}, is_training=False, n_classes=self.n_classes)
        
        with tf.compat.v1.variable_scope('', reuse=tf.compat.v1.AUTO_REUSE):
            self.net_150 = PGNModel({'data': image_batch150}, is_training=False, n_classes=self.n_classes)
        
        with tf.compat.v1.variable_scope('', reuse=tf.compat.v1.AUTO_REUSE):
            self.net_175 = PGNModel({'data': image_batch175}, is_training=False, n_classes=self.n_classes)
    
    def load(self, loader, sess, checkpoint_dir):
        """Load the pre-trained model."""
        checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            checkpoint_name = os.path.basename(checkpoint.model_checkpoint_path)
            loader.restore(sess, os.path.join(checkpoint_dir, checkpoint_name))
            return True
        else:
            return False
    
    def inference(self, storage_root: str, img_name: str):
        """Create the model and start the evaluation process."""
        # Create queue coordinator.
        coord = tf.train.Coordinator()
        
        # Load reader.
        with tf.name_scope("create_inputs"):
            reader = test_image_reader.TestImageReaderCustom(osp.join(storage_root, "raw_data/person", img_name), None, None, None, False, False, False, coord)
            image = reader.image

            # TODO: image를 정해진 크기로 resize(height: 256으로 고정해서)
            h, w = tf.shape(image)[0], tf.shape(image)[1]
            new_h = 256 # height를 256으로 고정
            new_w = tf.to_int32(tf.multiply(w, tf.divide(new_h, tf.to_float(h))))
            image = tf.image.resize_images(image, [new_h, new_w])

            image_rev = tf.reverse(image, tf.stack([1]))
            image_list = reader.image_list
        
        image_batch = tf.stack([image, image_rev])

        # parsing net
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            parsing_out1_100 = self.net_100.layers['parsing_fc']
            parsing_out2_100 = self.net_100.layers['parsing_rf_fc']
            edge_out2_100 = self.net_100.layers['edge_rf_fc']
        
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            parsing_out1_050 = self.net_050.layers['parsing_fc']
            parsing_out2_050 = self.net_050.layers['parsing_rf_fc']
        
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            parsing_out1_075 = self.net_075.layers['parsing_fc']
            parsing_out2_075 = self.net_075.layers['parsing_rf_fc']
        
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            parsing_out1_125 = self.net_125.layers['parsing_fc']
            parsing_out2_125 = self.net_125.layers['parsing_rf_fc']
            edge_out2_125 = self.net_125.layers['edge_rf_fc']
        
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            parsing_out1_150 = self.net_150.layers['parsing_fc']
            parsing_out2_150 = self.net_150.layers['parsing_rf_fc']
            edge_out2_150 = self.net_150.layers['edge_rf_fc']
        
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            parsing_out1_175 = self.net_175.layers['parsing_fc']
            parsing_out2_175 = self.net_175.layers['parsing_rf_fc']
            edge_out2_175 = self.net_175.layers['edge_rf_fc']

        # parsing_out1, parsing_out2, edge_out2를 사용하여 후처리 및 최종 결과 생성
        parsing_out1 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out1_050, tf.shape(image_batch)[1:3,]),
                                                tf.image.resize_images(parsing_out1_075, tf.shape(image_batch)[1:3,]),
                                                tf.image.resize_images(parsing_out1_100, tf.shape(image_batch)[1:3,]),
                                                tf.image.resize_images(parsing_out1_125, tf.shape(image_batch)[1:3,]),
                                                tf.image.resize_images(parsing_out1_150, tf.shape(image_batch)[1:3,]),
                                                tf.image.resize_images(parsing_out1_175, tf.shape(image_batch)[1:3,])]), axis=0)

        parsing_out2 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out2_050, tf.shape(image_batch)[1:3,]),
                                                tf.image.resize_images(parsing_out2_075, tf.shape(image_batch)[1:3,]),
                                                tf.image.resize_images(parsing_out2_100, tf.shape(image_batch)[1:3,]),
                                                tf.image.resize_images(parsing_out2_125, tf.shape(image_batch)[1:3,]),
                                                tf.image.resize_images(parsing_out2_150, tf.shape(image_batch)[1:3,]),
                                                tf.image.resize_images(parsing_out2_175, tf.shape(image_batch)[1:3,])]), axis=0)


        edge_out2_100 = tf.image.resize_images(edge_out2_100, tf.shape(self.image_placeholder)[1:3,])
        edge_out2_125 = tf.image.resize_images(edge_out2_125, tf.shape(self.image_placeholder)[1:3,])
        edge_out2_150 = tf.image.resize_images(edge_out2_150, tf.shape(self.image_placeholder)[1:3,])
        edge_out2_175 = tf.image.resize_images(edge_out2_175, tf.shape(self.image_placeholder)[1:3,])
        edge_out2 = tf.reduce_mean(tf.stack([edge_out2_100, edge_out2_125, edge_out2_150, edge_out2_175]), axis=0)

        raw_output = tf.reduce_mean(tf.stack([parsing_out1, parsing_out2]), axis=0)
        head_output, tail_output = tf.unstack(raw_output, num=2, axis=0)
        tail_list = tf.unstack(tail_output, num=20, axis=2)
        tail_list_rev = [None] * 20
        for xx in range(14):
            tail_list_rev[xx] = tail_list[xx]
        tail_list_rev[14] = tail_list[15]
        tail_list_rev[15] = tail_list[14]
        tail_list_rev[16] = tail_list[17]
        tail_list_rev[17] = tail_list[16]
        tail_list_rev[18] = tail_list[19]
        tail_list_rev[19] = tail_list[18]
        tail_output_rev = tf.stack(tail_list_rev, axis=2)
        tail_output_rev = tf.reverse(tail_output_rev, tf.stack([1]))

        raw_output_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
        raw_output_all = tf.expand_dims(raw_output_all, dim=0)
        self.pred_scores = tf.reduce_max(raw_output_all, axis=3)
        raw_output_all = tf.argmax(raw_output_all, axis=3)
        self.pred_all = tf.expand_dims(raw_output_all, dim=3)  # Create 4-d tensor.

        raw_edge_all = tf.reduce_mean(tf.stack([edge_out2]), axis=0)
        raw_edge_all = tf.expand_dims(raw_edge_all, dim=0)
        self.pred_edge = tf.sigmoid(raw_edge_all)
        res_edge = tf.cast(tf.greater(self.pred_edge, 0.5), tf.int32)

        # 나머지 코드는 이전과 동일하게 유지
        # Which variables to load.
        restore_var = tf.global_variables()

        # Set up tf session and initialize variables.
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        init = tf.global_variables_initializer()

        sess.run(init)
        sess.run(tf.local_variables_initializer())

        # Load weights.
        loader = tf.compat.v1.train.Saver(var_list=restore_var)
        if self.restore_from is not None:
            if load(loader, sess, self.restore_from):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
                return

        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        # evaluate prosessing
        parsing_dir = osp.join(storage_root, "preprocess/human_parse")
        if not osp.exists(parsing_dir):
            os.makedirs(parsing_dir)
        # edge_dir = './output/cihp_edge_maps'
        # if not osp.exists(edge_dir):
        #     os.makedirs(edge_dir)

        # Iterate over training steps.
        for step in range(self.num_steps):
            print(step)
            image_batch_np = sess.run(image_batch)
            parsing_, scores, edge_ = sess.run([self.pred_all, self.pred_scores, self.pred_edge], feed_dict={self.image_placeholder: image_batch_np})
            if step % 1 == 0:
                print('step {:d}'.format(step))
                print(image_list[step])
            img_id = img_name.split('.')[0]

            msk = decode_labels(parsing_, num_classes=self.n_classes)

            parsing_im = Image.fromarray(msk[0])

            # TODO: parsing_img를 원본 이미지 크기로 resize해서 저장.
            # parsing_im.resize((w, h), )
            parsing_im.save('{}/{}_vis.png'.format(parsing_dir, img_id))
            cv2.imwrite('{}/{}.png'.format(parsing_dir, img_id), parsing_[0, :, :, 0])
            # cv2.imwrite('{}/{}.png'.format(edge_dir, img_id), edge_[0, :, :, 0] * 255)
            print("here")

        coord.request_stop()
        coord.join(threads)

        save_state = False
        if osp.exists(osp.join(parsing_dir, img_id + '.png')) and osp.getsize(osp.join(parsing_dir, img_id + '.png')):
            save_state = True

        tf.reset_default_graph()

        return save_state
