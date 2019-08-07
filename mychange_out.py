# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 20:07:39 2018
pb模型测试程序
@author: Albert
"""

import tensorflow as tf
import numpy as np
import cv2
 
#模型路径
MODEL_PATH = 'models/cubist.pb'
#测试图片
image_dir = 'test_image/4.jpg'

def  test(model_path, testImage):
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()
        with open(model_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")
     
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            # x_test = x_test.reshape(1, 28 * 28)
            input_x = sess.graph.get_tensor_by_name("img_input:0")
            output = sess.graph.get_tensor_by_name("result_output:0")
     
            #对图片进行测试

            one_W, one_H = 500, 500

            styImgReaded = cv2.imread(testImage)
            styImgReaded = cv2.resize(styImgReaded, (one_W, one_H), interpolation=cv2.INTER_CUBIC)
            styImgReaded = np.expand_dims(styImgReaded, axis=0)

            #print(test_input)
            res = sess.run(output, feed_dict={input_x: styImgReaded})#利用训练好的模型预测结果
            print('模型运算完成')

            res = cv2.resize(res[0], (500, 500), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite('style_image/4.jpg', res)  
            print('over2')

            return

if __name__ == "__main__":
    test(model_path=MODEL_PATH, testImage=image_dir)
