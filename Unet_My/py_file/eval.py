'''
 * @author [Zizhao Zhang]
 * @email [zizhao@cise.ufl.edu]
 * @create date 2017-05-25 02:20:32
 * @modify date 2017-05-25 02:20:32
 * @desc [description]
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os, sys
import numpy as np
import scipy.misc as misc
from model import UNet
from utils import dice_coef, dice_coef_loss
from loader import dataLoader, folderLoader
from PIL import Image
from utils import VIS, mean_IU
import cv2
# configure args
from opts import *
# assert(opt.load_from_checkpoint != '')
# assert(opt.batch_size == 1)
from matplotlib import pyplot as plt 
#vis = VIS(save_path=opt.load_from_checkpoint)
vis = VIS(save_path='./trainlog')
# configuration session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# define data loader
img_shape = [ opt.imHeight,opt.imWidth]
test_generator, test_samples = folderLoader(opt.data_path, imSize=img_shape)
 
 

# define model, the last dimension is the channel
label = tf.placeholder(tf.int32, shape=[None]+img_shape)
with tf.name_scope('unet'):
    model = UNet().create_model(img_shape=img_shape+[3], num_class=opt.num_class)
    img = model.input
    pred = model.output
# define loss
with tf.name_scope('cross_entropy'): 
    cross_entropy_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=pred))

saver = tf.train.Saver() # must be added in the end

''' Main '''
init_op = tf.global_variables_initializer()
sess.run(init_op)
with sess.as_default():
    
    # restore from a checkpoint if exists
    try:
        
#        saver.restore(sess, './trainlog/')
#        saver.restore(sess, opt.load_from_checkpoint)
        module_file = tf.train.latest_checkpoint(opt.load_from_checkpoint)    
        saver.restore(sess, module_file)
        print ('--> load from checkpoint '+opt.load_from_checkpoint)
    except:
            print ('unable to load checkpoint ...')
            sys.exit(0)
    # debug
    dice_score = 0
    for it in range(0, test_samples):
        x_batch, y_batch, name = next(test_generator)
        # tensorflow wants a different tensor order
        feed_dict = {   
                        img: x_batch,
                        label: y_batch
                    }
        loss, pred_logits = sess.run([cross_entropy_loss, pred], feed_dict=feed_dict)
        pred_map = np.argmax(pred_logits[0],axis=2)
        score = vis.add_sample(pred_map, y_batch[0])
        print ('[iter %f]: loss=%f, meanIU=%f' % (it, loss, score))
        names='iter_'+str(it)+'loss_'+str(loss) +'meanIU_'+str(score)+'.bmp'
        cv2.imwrite(names,pred_map*255)
#         plt.imshow(pred_map)
#        vis.save_seg(pred_map, name='{0:}_{1:.3f}.png'.format(name, score), im=x_batch[0], gt=y_batch[0])

    vis.compute_scores()