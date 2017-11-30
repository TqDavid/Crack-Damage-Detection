# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 11:35:30 2017
数据增强、分离测试验证集
@author: jack
"""

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import os
import glob
import cv2
#from libtiff import TIFF


datapath='./data/'

class myAugmentation(object):
	
	"""
	A class used to augmentate image
	Firstly, read train image and label seperately, and then merge them together for the next process
	Secondly, use keras preprocessing to augmentate image
	Finally, seperate augmentated image apart into train image and label
	"""

	def __init__(self,  
              merge_path="./data/merge", 
              aug_merge_path="./data/aug_merge",
              aug_train_path="./data/aug_train",
              aug_label_path="./data/aug_label",
              val_img_path="./data/val/aug_train",
              val_label_path="./data/val/aug_label",
              img_type="png",
              splitRatio=0.8):
		
	 

		self.merge_imgs = glob.glob(merge_path+"/*."+img_type)		 
		self.merge_path = merge_path
		self.img_type = img_type
		self.splitRatio = splitRatio
		self.aug_merge_path = aug_merge_path
		self.aug_train_path = aug_train_path
		self.aug_label_path = aug_label_path
		self.val_img_path = val_img_path
		self.val_label_path = val_label_path
		self.slices = len(self.merge_imgs)
		self.datagen = ImageDataGenerator(
							        rotation_range=0.2,
							        width_shift_range=0.05,
							        height_shift_range=0.05,
							        shear_range=0.05,
							        zoom_range=0.05,
							        horizontal_flip=True,
							        fill_mode='nearest')

	def Augmentation(self):

		"""
		Start augmentation.....
		"""     
		     
		
		path_merge = self.merge_path

		print(path_merge)
		imgtype = self.img_type
		path_aug_merge = self.aug_merge_path
		if not os.path.lexists(path_aug_merge):
				os.mkdir(path_aug_merge)		
		for i in range(self.slices):
			
			 
			img=cv2.imread(self.merge_imgs[i])
			 
			img = img.reshape((1,) + img.shape)
			savedir = path_aug_merge + "/" + str(i)
			if not os.path.lexists(savedir):
				os.mkdir(savedir)
			self.doAugmentate(img, savedir, str(i))


	def doAugmentate(self, img, save_to_dir, save_prefix, batch_size=1, save_format='png', imgnum=30):

		"""
		augmentate one image
		"""
		datagen = self.datagen
		i = 0
		for batch in datagen.flow(img,
                          batch_size=batch_size,
                          save_to_dir=save_to_dir,
                          save_prefix=save_prefix,
                          save_format=save_format):
		    i += 1
		    if i > imgnum:
		        break

	def Merge(self):
		srcpath='./data/RoadCrack/'
		data_type='jpg'
		img_type='png'        
		path_merge = self.merge_path
		train_imgs= glob.glob(srcpath+"/*."+data_type)
		if not os.path.lexists(path_merge):
				os.mkdir(path_merge)
		for imgname in train_imgs:			
			img = cv2.imread(imgname,cv2.IMREAD_GRAYSCALE) 
			labelname = imgname[:imgname.rindex(data_type)]+img_type
			label = cv2.imread(labelname,cv2.IMREAD_GRAYSCALE)
			imgmerge = np.ndarray((img.shape[0],img.shape[1],3), dtype=np.uint8)
			imgmerge[:,:,0] = label
			imgmerge[:,:,1] = label
			imgmerge[:,:,2] = img
			midname = imgname[imgname.rindex("\\")+1:imgname.rindex("."+data_type)]
			cv2.imwrite(path_merge+"/"+midname+"."+img_type,imgmerge) 
          
#			cv2.imshow('img',img)
#			cv2.imshow('imgmerge',imgmerge)
#			cv2.waitKey(1)
#		for i in range(self.slices):
#			 
	def splitMerge(self):

		"""		merged Src data
		"""
		img_type='png'
		path_merge = self.aug_merge_path
		path_train = self.aug_train_path
		path_label = self.aug_label_path
		if not os.path.lexists(path_train):
				os.mkdir(path_train)
		if not os.path.lexists(path_label):
				os.mkdir(path_label)
		if not os.path.lexists(self.val_img_path):
				os.mkdir(self.val_img_path)
		if not os.path.lexists(self.val_label_path):
				os.mkdir(self.val_label_path)
		for i in range(self.slices):
			if i<self.slices*self.splitRatio:
				path = path_merge + "/" + str(i)
				train_imgs = glob.glob(path+"/*."+img_type)
				print('here is %d train imgs',len(train_imgs))
				savedir = path_train + "/" + str(i)
				if not os.path.lexists(savedir):
				   os.mkdir(savedir)
				savedir = path_label + "/" + str(i)
				if not os.path.lexists(savedir):
				   os.mkdir(savedir)
				for imgname in train_imgs:
				   midname = imgname[imgname.rindex("\\")+1:imgname.rindex("."+img_type)]
				   img = cv2.imread(imgname)
    #				print(img)
				   img_train = img[:,:,0]#cv2 read image rgb->bgr
				   img_label = img[:,:,2]				 
				   cv2.imwrite(path_train+"/"+str(i)+"\\"+midname+"_train"+"."+img_type,img_train)
				   cv2.imwrite(path_label+"/"+str(i)+"\\"+midname+"_label"+"."+img_type,img_label)
			else:
				path = path_merge + "/" + str(i)
				train_imgs = glob.glob(path+"/*."+img_type)
				print('here is %d train imgs',len(train_imgs))
				savedir = self.val_img_path + "/" + str(int(i-self.slices*self.splitRatio))
				if not os.path.lexists(savedir):
				   os.mkdir(savedir)
				savedir = self.val_label_path + "/" + str(int(i-self.slices*self.splitRatio))
				if not os.path.lexists(savedir):
				   os.mkdir(savedir)
				for imgname in train_imgs:
				   midname = imgname[imgname.rindex("\\")+1:imgname.rindex("."+img_type)]
				   img = cv2.imread(imgname)
    #				print(img)
				   img_train = img[:,:,0]#cv2 read image rgb->bgr
				   img_label = img[:,:,2]				 
				   cv2.imwrite(self.val_img_path+"/"+str(int(i-self.slices*self.splitRatio))+"\\"+midname+"_train"+"."+img_type,img_train)
				   cv2.imwrite( self.val_label_path+"/"+str(int(i-self.slices*self.splitRatio))+"\\"+midname+"_label"+"."+img_type,img_label)

	 



class dataProcess(object):

	def __init__(self, out_rows, out_cols, data_path = "./data/aug_train/0", label_path = "./data/aug_label/0", test_path = "./data/test", npy_path = "../npydata", img_type = "png"):

		"""		
		"""

		self.out_rows = out_rows
		self.out_cols = out_cols
		self.data_path = data_path
		self.label_path = label_path
		self.img_type = img_type
		self.test_path = test_path
		self.npy_path = npy_path

	def create_train_data(self):
		i = 0
		print('-'*30)        
		print('Creating training images...')		 
		print('-'*30)
		imgs = glob.glob(self.data_path+"/*."+self.img_type)		 
		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		imglabels = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		for imgname in imgs: 
         
			midname = imgname[imgname.rindex("\\")+1:]
#			print('midname')
#			print(midname)
#			print('self.data_path + "/" + midname')
#			print(self.data_path + "/" + midname)
#			img = load_img(self.data_path + "/" + midname,grayscale = True)
#			midname=midname[:midname.rindex('train')]
#			label = load_img(self.label_path + "/" + midname+'label.png',grayscale = True)
#			
#			img = img_to_array(img)
#			
#			label = img_to_array(label)
			img = cv2.imread(self.data_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
			 
			midname=midname[:midname.rindex('train')]
			label = cv2.imread(self.label_path + "/" + midname+'label.png',cv2.IMREAD_GRAYSCALE)
			img=cv2.resize(img,(self.out_rows,self.out_cols))
			label=cv2.resize(label,(self.out_rows,self.out_cols))
			 
#			img = np.array([img])
			img=img.reshape((self.out_rows,self.out_cols,1))
#			label = np.array([label])
			label=label.reshape((self.out_rows,self.out_cols,1))
			imgdatas[i] = img
			imglabels[i] = label
			if i % 100 == 0:
				print('Done: {0}/{1} images'.format(i, len(imgs)))
			i += 1
		print('loading done')
		np.save(self.npy_path + '/imgs_train.npy', imgdatas)
		np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
		print('Saving to .npy files done.')

	def create_test_data(self):
		i = 0
		print('-'*30)
		print('Creating test images...')
		print('-'*30)
		imgs = glob.glob(self.test_path+"/*."+'tif')		 
		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		for imgname in imgs:
			print(imgname)
			 
#			img = load_img(self.test_path + "/" + midname,grayscale = True)
#			img = img_to_array(img)
			img = cv2.imread(imgname,cv2.IMREAD_GRAYSCALE)
			img=cv2.resize(img,(self.out_rows,self.out_cols))		  
#			img = np.array([img])
			img=img.reshape((self.out_rows,self.out_cols,1))
#			label = np.array([label])
			 
			imgdatas[i] = img
			i += 1
		print('loading done')
		np.save(self.npy_path + '/imgs_test.npy', imgdatas)
		print('Saving to imgs_test.npy files done.')

	def load_train_data(self):
		print('-'*30)
		print('load train images...')
		print('-'*30)
		imgs_train = np.load(self.npy_path+"/imgs_train.npy")
		imgs_mask_train = np.load(self.npy_path+"/imgs_mask_train.npy")
		imgs_train = imgs_train.astype('float32')
		imgs_mask_train = imgs_mask_train.astype('float32')
		imgs_train /= 255
		#mean = imgs_train.mean(axis = 0)
		#imgs_train -= mean	
		imgs_mask_train /= 255
		imgs_mask_train[imgs_mask_train > 0.5] = 1
		imgs_mask_train[imgs_mask_train <= 0.5] = 0
		return imgs_train,imgs_mask_train

	def load_test_data(self):
		print('-'*30)
		print('load test images...')
		print('-'*30)
		imgs_test = np.load(self.npy_path+"/imgs_test.npy")
		imgs_test = imgs_test.astype('float32')
		imgs_test /= 255
		#mean = imgs_test.mean(axis = 0)
		#imgs_test -= mean	
		return imgs_test

if __name__ == "__main__":

 	aug = myAugmentation()
 	aug.Merge()
 	aug.Augmentation()
 	aug.splitMerge()
 
#	mydata = dataProcess(256,256)
#	mydata.create_train_data()
#	mydata.create_test_data()
#	imgs_train,imgs_mask_train = mydata.load_train_data()
#	print (imgs_train.shape,imgs_mask_train.shape)
