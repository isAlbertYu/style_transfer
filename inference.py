# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 19:52:56 2019

@author: Administrator
"""
import tensorflow as tf 
import tensorflow.examples.tutorials.mnist.input_data as input_data

from tensorflow.python.platform import gfile


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)     #下载并加载mnist数据


sess = tf.Session()

sess.graph.as_default()
    
    
f = gfile.FastGFile('models/wave.pb', "rb")
gd = graph_def = tf.GraphDef()
graph_def.ParseFromString(f.read())

# fix nodes
for node in graph_def.node:
    if node.op == 'RefSwitch':
        node.op = 'Switch'
        for index in range(len(node.input)):
            if 'moving_' in node.input[index]:
                node.input[index] = node.input[index] + '/read'
    elif node.op == 'AssignSub':
        node.op = 'Sub'
        if 'use_locking' in node.attr: del node.attr['use_locking']

# import graph into session
tf.import_graph_def(graph_def, name='')


print('here------1')
tf.import_graph_def(graph_def, name='') # 导入计算图
print('here------2')
# 需要有一个初始化的过程
sess.run(tf.global_variables_initializer())
#for op in sess.graph.get_operations():
#    print(op.name)
# 需要先复原变量
x_in = sess.graph.get_tensor_by_name('input:0')
y_out = sess.graph.get_tensor_by_name('output:0')
keep_prob = sess.graph.get_tensor_by_name('keep_prob:0')


y_predict = sess.run(y_out, feed_dict={x_in: mnist.test.images, keep_prob:1})



correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(mnist.test.labels,1))    
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))                 #精确度计算
print('accuracy= ', sess.run(accuracy))

