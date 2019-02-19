# -*- coding: utf-8 -*-

import tensorflow as tf
import cv2, model, data_io
import numpy as np

image_file_dir = 'D:\\MyProgramma\\myPy\\21_gun\\7_gun\\my\\train2014'
image_dir = 'D:\\MyProgramma\\myPy\\21_gun\\7_gun\\my\\mymodel2\\test.jpg'
BATCH_SIZE = 1
num_epochs = 1
IMG_W, IMG_H = 256, 256

#ContImgReaded = data_io.get_cont_batch(image_file_dir, IMG_W, IMG_H, BATCH_SIZE, num_epochs)#---喂给content_images
styImgReaded = np.expand_dims(cv2.imread(image_dir), axis=0)#---喂给style_image
styImgReadedsize = styImgReaded.shape
styImg_W = styImgReadedsize[1]
styImg_H = styImgReadedsize[2]


content_images = tf.placeholder(tf.float32, shape=[BATCH_SIZE, styImg_W, styImg_H, 3])

generated_image = model.net(content_images, training=True)


saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)


    print('in')
    
    saver.restore(sess, tf.train.latest_checkpoint('D:\\MyProgramma\\myPy\\21_gun\\7_gun\\my\\15000save\\'))
    print('over')

    res = sess.run(generated_image, feed_dict={content_images:styImgReaded})
    cv2.imwrite('D:\\MyProgramma\\myPy\\21_gun\\7_gun\\my\\mymodel2//res'+str(2)+'.jpg', res[0])
    
    print('over2')
    coord.join(threads)
    

