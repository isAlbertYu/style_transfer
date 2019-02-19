
import tensorflow as tf
import tools
import numpy as np
#%%
'''
isSty:是否在计算风格图像
'''
def VGG16(x, isSty, is_pretrain=False):
    x = x - np.array([ 123.68 ,  116.779,  103.939])
    every_layer_output = {}
    
    x = tools.conv('conv1_1', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    
    x = tools.conv('conv1_2', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    every_layer_output['conv1_2'] = x #conv1_2
    x = tools.pool('pool1', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
    
    x = tools.conv('conv2_1', x, 128, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    
    x = tools.conv('conv2_2', x, 128, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    every_layer_output['conv2_2'] = x #conv2_2
    
    if isSty:
        x = tools.pool('pool2', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
        
        x = tools.conv('conv3_1', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        
        x = tools.conv('conv3_2', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        x = tools.conv('conv3_3', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        every_layer_output['conv3_3'] = x #conv3_3
    
        
        x = tools.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
        
        x = tools.conv('conv4_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        
        x = tools.conv('conv4_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        
        x = tools.conv('conv4_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        every_layer_output['conv4_3'] = x #conv4_3

#    x = tools.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
#
#    x = tools.conv('conv5_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
    
#    x = tools.conv('conv5_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
#    x = tools.conv('conv5_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
#    x = tools.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)            
#
#    x = tools.FC_layer('fc6', x, out_nodes=4096)
#    #x = tools.batch_norm(x)
#    x = tools.FC_layer('fc7', x, out_nodes=4096)
#    #x = tools.batch_norm(x)
#    x = tools.FC_layer('fc8', x, out_nodes=n_classes)

    return every_layer_output
        




#%% TO get better tensorboard figures!

def VGG16N(x, n_classes, is_pretrain=True):
    
    with tf.name_scope('VGG16'):

        x = tools.conv('conv1_1', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)   
        x = tools.conv('conv1_2', x, 64, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        with tf.name_scope('pool1'):    
            x = tools.pool('pool1', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
            
        x = tools.conv('conv2_1', x, 128, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)    
        x = tools.conv('conv2_2', x, 128, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        with tf.name_scope('pool2'):    
            x = tools.pool('pool2', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
         
            

        x = tools.conv('conv3_1', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        x = tools.conv('conv3_2', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        x = tools.conv('conv3_3', x, 256, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        with tf.name_scope('pool3'):
            x = tools.pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
            

        x = tools.conv('conv4_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        x = tools.conv('conv4_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        x = tools.conv('conv4_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        with tf.name_scope('pool4'):
            x = tools.pool('pool4', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
        

        x = tools.conv('conv5_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        x = tools.conv('conv5_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        x = tools.conv('conv5_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1], is_pretrain=is_pretrain)
        with tf.name_scope('pool5'):
            x = tools.pool('pool5', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)            
        
        
        x = tools.FC_layer('fc6', x, out_nodes=4096)        
        with tf.name_scope('batch_norm1'):
            x = tools.batch_norm(x)           
        x = tools.FC_layer('fc7', x, out_nodes=4096)        
        with tf.name_scope('batch_norm2'):
            x = tools.batch_norm(x)            
        x = tools.FC_layer('fc8', x, out_nodes=n_classes)
    
        return x



#%%







            
