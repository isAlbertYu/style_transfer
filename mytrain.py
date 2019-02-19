# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import tools, data_io, transform, myutil, cv2, os, model
from scipy import misc

IMG_W = 256
IMG_H = 256
BATCH_SIZE = 1
learning_rate = 0.001
num_epochs = 2
## Weight of the loss
content_weight = 1.0  # weight for content features loss
style_weight = 100.0  # weight for style features loss
tv_weight = 200

STYLE_LAYERS = ('conv1_2', 'conv2_2', 'conv3_3', 'conv4_3')
CONTENT_LAYER = 'conv2_2'

def train():
    
    pre_trained_weights = './/vgg16.npy'
    style_image_dir = 'D:\\MyProgramma\\myPy\\21_gun\\7_gun\\my\\style_image\\mosaic.jpg'
    image_file_dir = 'D:\\MyProgramma\\myPy\\21_gun\\7_gun\\my\\train2014'
    save_dir = './/myModel//'
    exp_dir = './/myGenerate'
    now_exp_dir = myutil.makeExpDir(exp_dir)
    
## 读取一张固定的风格图像(实际数据) 
    #styImgReaded = data_io.get_style_batch(style_image_dir)
    styImgReaded = np.expand_dims(misc.imread(style_image_dir), axis=0)#---喂给style_image
    styImgReadedsize = styImgReaded.shape
    styImg_W = styImgReadedsize[1]
    styImg_H = styImgReadedsize[2]
    

## 读取一批内容图像（是通道，没有实际数据）
    ContImgReaded = data_io.get_cont_batch(image_file_dir, 256, 256, BATCH_SIZE, num_epochs)#---喂给content_images

                    
## 一张固定的风格图像 通道
    style_image = tf.placeholder(tf.float32, shape=[1, styImg_W, styImg_H, 3]) 

## 一批内容图像 的某层特征图 通道
    content_images = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    
##一批风格转换网络的生成图像
#    generated_image = transform.net(content_images)
    generated_image = model.net(content_images, training=True)
    
## 风格损失
    style_gram_dict = myutil.img2Gram(style_image)#风格图像的Gram矩阵字典
    gen_gram_dict = myutil.img2Gram(generated_image)#生成图像的Gram矩阵字典
    style_loss = tools.style_loss(gen_gram_dict, style_gram_dict, style_weight)

## 内容损失
    content_map_dict = myutil.img2ContMap(content_images)#内容图像的内容特征字典
    generated_feature_maps = myutil.img2ContMap(generated_image)#生成图像的内容特征字典
    content_loss = tools.content_loss(generated_feature_maps[CONTENT_LAYER], content_map_dict[CONTENT_LAYER], content_weight)

## tv损失
#    tv_loss = tools.tv_loss(generated_image, tv_weight)
      
## 总损失
#    loss = content_loss + style_loss + tv_loss
    loss = content_loss + style_loss
## 变学习率   
    global_step = tf.Variable(0, trainable=False)

    learning_rate = tf.train.exponential_decay(learning_rate=1e-3, global_step=global_step,
                                           decay_steps=300, decay_rate=0.98, staircase=True)
    
## 训练操作    
    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, global_step=global_step)
    
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # 加载vgg参数，跳过list中的参数不加载
        tools.load_with_skip(pre_trained_weights, sess, ['conv5_1','conv5_2','conv5_3','fc6','fc7','fc8'])  

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print('in')
        img_index = 0
        try:        
            while True:
                img_index += 1
                
                #获取一批内容图像
                ContImgBatch = sess.run(ContImgReaded)
                       
                ## 计算风格图像 的各层特征图的Gram矩阵
                sess.run(train_op, feed_dict={style_image:styImgReaded, content_images:ContImgBatch})
                                     
                if img_index % 10 == 0:
                    tra_loss, res, now_lr = sess.run([loss,generated_image,learning_rate], feed_dict={style_image:styImgReaded, content_images:ContImgBatch})
                    misc.imsave(now_exp_dir+'/res'+str(img_index)+'.jpg', res[0])
                    
                    print ('img_index: %d, loss: %.4f, lr: %.3f*e-3' % (img_index, tra_loss, now_lr*1000))
                    

                if img_index % 100 == 0:#保存模型
                    checkpoint_path = os.path.join(save_dir, 'myModel.ckpt')
                    saver.save(sess, checkpoint_path, global_step=img_index)
                    
                print('img_index : ', img_index)
        
        except tf.errors.OutOfRangeError:
            print("done")
     
        finally:
            coord.request_stop()        
            
                        
        coord.join(threads)
        sess.close()

train()
#InternalError: cuDNN Backward Filter function launch failure : input shape([1,64,391,470]) filter shape([3,3,64,64])
'''
exp01:
    将内容图像直接缩放成统一规格256*256
    可能会将许多细节部分的信息丢失，使得生成网络对细节内容难以复现
    改进:
        将内容图像裁剪成256*256
        然后测试时，不再裁剪，直接上原图来进入生成网络

'''


