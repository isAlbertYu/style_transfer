import tensorflow as tf
import glob



def get_files(image_file_dir):
    '''
    Args:
        file_dir: 图像文件夹
    Returns:
        图像路径列表
    '''
    image_dirs_list = glob.glob(image_file_dir+'\\*.jpg') 

    return image_dirs_list


def get_cont_batch(image_file_dir, image_W, image_H, batch_size, num_epochs):
    '''
    Args:
        num_epochs: 数据输出的轮数
    Returns:
        image_batch: [batch_size, width, height, 3]
    '''
    image_dirs_list = get_files(image_file_dir)
    image = tf.cast(image_dirs_list, tf.string)

    # make an input queue
    input_queue = tf.train.slice_input_producer([image], 
                                                num_epochs=num_epochs,#num_epochs为每个数据输出的次数，若num_epochs=None则循环输出无限次
                                                shuffle=False)
    
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents, channels=3)

##-----
    image_resized = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
#    image = tf.image.resize_images(image, [image_W, image_H])
#    image = tf.image.per_image_standardization(image)
    image_batch_resized = tf.train.batch([image_resized], batch_size= batch_size,
                                 num_threads= 64, capacity = 50)
    image_batch_resized = tf.cast(image_batch_resized, tf.float32)

##-----
#    image_batch = tf.train.batch([image], batch_size= batch_size,
#                                 num_threads= 64, capacity = 50)
#    image_batch = tf.cast(image_batch, tf.float32)
    
    
    return image_batch_resized


def get_style_batch(image_dir):
    image_value = tf.read_file(image_dir)
    img = tf.image.decode_jpeg(image_value, channels=3)
    img_batch = tf.expand_dims(img, dim=0)
    img_batch = tf.cast(img_batch, dtype=tf.float32)
    return img_batch
    
    
#def preprocess(image):
#    MEAN_PIXEL = np.array([ 123.68 ,  116.779,  103.939])
#    return image - MEAN_PIXEL


#image_file_dir = 'D:\\MyProgramma\\myPy\\TF_Google\\CatVsDog\\dataset\\11'
#ContImgReaded = get_cont_batch(image_file_dir, 256, 256, 4, 1)
#
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    sess.run(tf.local_variables_initializer())
#    # 加载vgg参数，跳过list中的参数不加载
##    tools.load_with_skip(pre_trained_weights, sess, ['conv5_2','conv5_3','fc6','fc7','fc8'])  
#
#    coord = tf.train.Coordinator()
#    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#    try:        
#        while True:
#            ContImgBatch = sess.run(ContImgReaded)
#    
#    except tf.errors.OutOfRangeError:
#        print("done")
# 
#    finally:
#        coord.request_stop()        
#        
#                    
#    coord.join(threads)
#    sess.close()
    
    
    

