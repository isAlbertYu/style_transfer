# -*- coding: utf-8 -*-

import tensorflow as tf
import tools, VGG, os


## 图像 -> 各层特征图 --> 各层特征图的伸展矩阵 -> 各层特征图的Gram矩阵
'''
style_image:一张固定的风格图像(占位符)
'''
def img2Gram(image):
    feature_map_dict = VGG.VGG16(image, isSty=True, is_pretrain=False)#图像在vgg各层的特征图字典
    gram_dict = {}
    STYLE_LAYERS = ('conv1_2', 'conv2_2', 'conv3_3', 'conv4_3')
    for layer in STYLE_LAYERS: 
        feature_map = feature_map_dict[layer]# feature_map.shape = (b, x, y , c)
        mapShape = tf.shape(feature_map)
        feature_map = tf.reshape(feature_map, (-1, mapShape[1]*mapShape[2], mapShape[3]))#将每层的特征图整成二维张量
        # feature_map.shape = (b , xy, c)
        gram = tools.Gram(feature_map)#计算二维张量特征图的Gram矩阵
        gram_dict[layer] = gram#各层的Gram矩阵字典
    return gram_dict

## 图像 -> 各层特征图
'''
image:
'''
def img2ContMap(image):
    feature_maps = VGG.VGG16(image, isSty=False, is_pretrain=False)#图像在vgg各层的特征图字典

    content_map_dict = {} 
    CONTENT_LAYER = 'conv2_2'
    content_map_dict[CONTENT_LAYER] = feature_maps[CONTENT_LAYER]#取'conv4_2'层的特征图，构成一个字典
    
    return content_map_dict
    

def makeExpDir(expdir):
    explist = os.listdir(expdir)
    num = len(explist) + 1
    if num < 10:
        dirname = expdir + '\\exp0' + str(num)
    else:
        dirname = expdir + '\\exp' + str(num)
    os.makedirs(dirname)
    
    return dirname

#style_image_dir = 'D:\\MyProgramma\\myPy\\21_gun\\7_gun\\my\\style_image\\feathers.jpg'
#
#styImgReaded = np.expand_dims(cv2.imread(style_image_dir), axis=0)
#styImgReadedsize = styImgReaded.shape
#styImg_W = styImgReadedsize[1]
#styImg_H = styImgReadedsize[2]
#
#style_image = tf.placeholder(tf.float32, shape=[1, styImg_W, styImg_H, 3])#一张固定的风格图像
#style_gram = img2Gram(style_image)
#
#pre_model = './/vgg16.npy'
#
#init = tf.global_variables_initializer()
#with tf.Session() as sess:
#    print('in')
#    sess.run(init)
#    tools.load_with_skip(pre_model, sess, ['conv5_2','conv5_3','fc6','fc7','fc8'])  
#    
#    a = sess.run([style_gram], feed_dict={style_image: styImgReaded})
#
#    print('over')


