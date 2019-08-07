# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import tools, VGG, data_io, transform, cv2

tf.reset_default_graph()

#%%
IMG_W = 256
IMG_H = 256
N_CLASSES = 2
BATCH_SIZE = 4
learning_rate = 0.001
MAX_STEP = 15000   # it took me about one hour to complete the training.
IS_PRETRAIN = False
CAPACITY = 2000

## Weight of the loss
content_weight = 1.0  # weight for content features loss
style_weight = 220.0  # weight for style features loss

STYLE_LAYERS = ('conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1')
CONTENT_LAYER = 'conv4_2'


pre_trained_weights = './/vgg16.npy'
style_image_dir = 'D:\\MyProgramma\\myPy\\21_gun\\7_gun\\my\\style_image\\feathers.jpg'
image_file_dir = 'D:\\MyProgramma\\myPy\\21_gun\\7_gun\\my\\train2014'


## 读取风格图像（是通道，没有实际数据）
#styImgReaded = data_io.get_style_batch(style_image_dir)
styImgReaded = np.expand_dims(cv2.imread(style_image_dir), axis=0)
styImgReadedsize = styImgReaded.shape
styImg_W = styImgReadedsize[1]
styImg_H = styImgReadedsize[2]

## 读取内容图像（是通道，没有实际数据）
ContImgReaded = data_io.get_cont_batch(image_file_dir, 256, 256, BATCH_SIZE)

                
## 风格图像 的各层特征图的Gram矩阵（是通道，没有实际数据） 
style_image = tf.placeholder(tf.float32, shape=[1, styImg_W, styImg_H, 3])#一张固定的风格图像
style_feature_maps = VGG.VGG16(style_image, is_pretrain=True)#风格图像经过vgg各层后的特征图字典
style_gram = {}

for feature_map in style_feature_maps.items(): 
    if feature_map[0] !=  CONTENT_LAYER:
        feature_map_value = feature_map[1]
        feature_map_value = tf.reshape(feature_map_value, (-1, tf.shape(feature_map_value)[1]*tf.shape(feature_map_value)[2], tf.shape(feature_map_value)[3]))#将每层的特征图整成二维张量

        gram = tools.Gram(feature_map_value)#计算二维张量特征图的Gram矩阵
        style_gram[feature_map[0]] = gram#各层的Gram矩阵字典
    

## 内容图像 的某层特征图的伸展矩阵
#content_image：待转换风格的图像
content_image = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3], name="content_image")
content_feature_maps = VGG.VGG16(content_image, is_pretrain=True)

content_features = {} #内容图像的一个特征图 字典
content_features[CONTENT_LAYER] = content_feature_maps[CONTENT_LAYER]

## 风格损失

generated_image = transform.net(content_image)#风格转换网络 生成图像
generated_feature_maps = VGG.VGG16(generated_image, is_pretrain=True) #生成图像的各个特征图

generated_gram = {} #一批图像各层特征图的gram矩阵
for feature_map in generated_feature_maps.items():
    if feature_map[0] != CONTENT_LAYER:
        feature_map_value = feature_map[1]
        feature_map_value = tf.reshape(feature_map_value, (-1, tf.shape(feature_map_value)[1]*tf.shape(feature_map_value)[2], tf.shape(feature_map_value)[3]))#将每层的特征图整成二维张量
    
        gram = tools.Gram(feature_map_value)  #计算二维张量特征图的Gram矩阵
        generated_gram[feature_map[0]] = gram #各层的Gram矩阵字典


style_loss = tools.style_loss(generated_gram, style_gram, style_weight)

## 内容损失
content_loss = tools.content_loss(generated_feature_maps[CONTENT_LAYER], content_features[CONTENT_LAYER], content_weight)
   
## 总损失
loss = content_loss + style_loss


train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    print('in')
    sess.run(init)
    # 加载vgg参数，跳过list中的参数不加载
    tools.load_with_skip(pre_trained_weights, sess, ['conv5_2','conv5_3','fc6','fc7','fc8'])  

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    
    ## 计算风格图像 的各层特征图的Gram矩阵
    style_gram_data = sess.run([style_gram], feed_dict={style_image:styImgReaded})
    
    
    print('over')

                
    coord.join(threads)
    sess.close()






