# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 13:24:22 2019

@author: ANDREA
"""

import os
import cv2 
import numpy as np

data='C:/Users/ANDREA/Desktop/PROGETTO PACI/flowers'
label=["daisy","dandelion","rose","sunflower","tulip"]
data_new=[]

#TO READ DATA
def fun():
    for i in label:
        path=os.path.join(data,i)
        clas=label.index(i)
        for s in os.listdir(path):
            try:
                array=cv2.imread(os.path.join(path,s),cv2.IMREAD_COLOR)
                array2=cv2.resize(array,(200,200))
                data_new.append([array2,clas])
            except Exception as e:
                pass

fun()


#SPLIT TRAIN AND TEST
from sklearn.model_selection import train_test_split

train,test=train_test_split(data_new,test_size=0.2,random_state=2)

#CREATE INPUT AND OUTPUT IN TRAIN AND TEST

X_train=[]
X_test=[]
Y_train=[]
Y_test=[]

#TRAIN
for pixel,label in train:
    X_train.append(pixel)
    Y_train.append(label)
#TEST
for pixel,label in test:
    X_test.append(pixel)
    Y_test.append(label)
    
#RESHAPING OF INPUT:
X_train=np.array(X_train)
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],3)

X_test=np.array(X_test)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],3)

#IDENTIFY THE CATEGORIES:
from keras.utils import to_categorical
Y_train=to_categorical(Y_train)

Y_test=to_categorical(Y_test)


#################################################################à
#BUILDING A CNN

#IMPORT PACKAGES AND LIBRARIES:
import keras
from keras import models
from keras import layers
import tensorflow as tf 
from keras.layers import Input,Flatten,Dense,Dropout
from keras.models import Model
from keras import optimizers
from keras.layers.normalization import BatchNormalization

#VGG19 CNN
from keras.applications.vgg19 import VGG19
def Vgg19():
    vgg19=VGG19(weights='imagenet',input_shape=X_train.shape[1:],include_top=False)
    for layer in vgg19.layers[:1]:
        a=vgg19.output
        a=Flatten()(a)
        a=Dense(1000,activation='relu')(a)
        a=Dropout(0.5)(a)
        a=Dense(1000,activation='relu')(a)
        pred=Dense(5,activation='softmax')(a)
        model=Model(inputs=vgg19.input,output=pred)
        model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
        return model
VGG19=Vgg19()
VGG19.summary()
keras2ascii(VGG19)



#LE NET5 CNN

def LeNet():
    lenet5=models.Sequential()
    lenet5.add(layers.Conv2D(6,(5,5),activation='tanh',input_shape=(32,32,1),padding='same'))
    lenet5.add(layers.AveragePooling2D((2,2),padding='valid'))
    lenet5.add(layers.Conv2D(16,(5,5),activation='tanh',strides=(1,1),padding='valid'))
    lenet5.add(layers.AveragePooling2D((2,2),padding='valid',strides=(2,2)))
    lenet5.add(layers.Conv2D(120,(5,5),activation='tanh',strides=(1,1),padding='valid'))
    lenet5.add(layers.Flatten())
    lenet5.add(layers.Dense(84,activation='tanh'))
    lenet5.add(layers.Dense(output_dim=5,activation='softmax'))
    lenet5.compile(optimizer='SGD',metrics=['accuracy'],loss='categorical_crossentropy')
    return lenet5
LENET5=LeNet()
LENET5.summary()

#RESNET 50
from keras.applications.resnet50 import ResNet50 

def Resnet50():
    res=models.Sequential()
    res.add(ResNet50(weights='imagenet',include_top=False,input_shape=X_train.shape[1:]))
    res.add(Flatten())
    res.add(BatchNormalization())
    res.add(Dense(2000,activation='relu'))
    res.add(BatchNormalization())
    res.add(Dense(5,activation='softmax'))
    res.compile(optimizer='adam',metrics=['accuracy'],loss='categorical_crossentropy')
    return res

RES50=Resnet50()
RES50.summary()

keras2ascii(RES50)

#ALEXNET

def Alexnet():
    alex = models.Sequential()
    alex.add(layers.Conv2D(filters=48, input_shape=X_train.shape[1:], kernel_size=(11,11), strides=(4,4), padding='same',activation='relu'))
    alex.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    alex.add(layers.Conv2D(96, kernel_size=(3,3),activation='relu', padding='same'))
    alex.add(layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))
    alex.add(layers.Conv2D(filters=192, kernel_size=(3,3), padding='same',activation='relu'))
    alex.add(layers.Conv2D(filters=192, kernel_size=(3,3), padding='same',activation='relu'))
    alex.add(layers.Conv2D(filters=128, kernel_size=(3,3), padding='same',activation='relu'))
    alex.add(layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    alex.add(layers.Flatten())
    alex.add(layers.Dense(2048, input_shape=(200*200*3,),activation='relu'))
    alex.add(layers.Dropout(0.4))
    alex.add(layers.Dense(2048,activation='relu'))
    alex.add(layers.Dropout(0.4))
    alex.add(layers.Dense(1000,activation='relu'))
    alex.add(layers.Dropout(0.4))
    alex.add(layers.Dense(5,activation='softmax'))
    return alex
    
ALEXNET=Alexnet()
ALEXNET.summary()

keras2ascii(ALEXNET)
    
    

##########################
#VISUALIZATION OF THE MODELS  
#IMPORT LIBRARIES
from keras_sequential_ascii import keras2ascii



keras2ascii(LENET5)


import pydot_ng as pydot 
pydot.find_graphviz
import graphviz 
from keras.utils.vis_utils import plot_model
plot_model(model,to_file='png',show_shapes=True,show_layer_names=True )
#FROM SCRATCH 
#ALEXNET 
#ZFNET
#INCEPTION OR GOOGLENET 

#DATA AUGMENTATION 
from keras.preprocessing.image import ImageDataGenerator
#For train images:
image_gen = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=.4,height_shift_range=.4,zoom_range=0.3,horizontal_flip=True)

image_gen.fit(X_train, augment=True)

image_gen_val=ImageDataGenerator(rescale=1./255)


#FITTING THE RESULTS:
#VGG
Vgg=VGG19.fit(X_train,Y_train,batch_size=20,epochs=30,validation_data=(X_test,Y_test),validation_steps=len(X_test)/32)
#WITH AUGMENTED DATA:
Vgg2=VGG19.fit_generator(image_gen.flow(X_train, Y_train, batch_size=32),epochs=30, validation_data=image_gen_val.flow(X_test,Y_test),validation_steps=len(X_test)/32)
#######################################
#PLOTTING THE RESULTS
import matplotlib.pyplot as plt

#ACCURACY

plt.plot(one.history['acc'],'bo',label='Training accuracy')
plt.plot(one.history['val_acc'],'b',label='Validation accuracy')
plt.legend()
plt.title('Accuracy in train and test')
plt.show()

#LOSS

plt.plot(one.history['loss'],'bo',label='Training loss')
plt.plot(one.history['val_loss'],'b',label='Validation loss')
plt.legend()
plt.title('Loss in train and test')
plt.show()

#BECAUSE OF THE HIGH OVERFITTING: DROPOUT



onedr=LENET5.fit(X_train,Y_train,batch_size=20,epochs=30,validation_data=(X_test,Y_test))


plt.plot(onedr.history['acc'],'bo',label='Training accuracy')
plt.plot(onedr.history['val_acc'],'b',label='Validation accuracy')
plt.legend()
plt.title('Accuracy in train and test')
plt.show()

#LOSS

plt.plot(onedr.history['loss'],'bo',label='Training loss')
plt.plot(onedr.history['val_loss'],'b',label='Validation loss')
plt.legend()
plt.title('Loss in train and test')
plt.show()
 



plt.plot(FIT.history['acc'],'bo',label='Training accuracy')
plt.plot(FIT.history['val_acc'],'b',label='Validation accuracy')
plt.legend()
plt.title('Accuracy in train and test')
plt.show()

#LOSS

plt.plot(FIT.history['loss'],'bo',label='Training loss')
plt.plot(FIT.history['val_loss'],'b',label='Validation loss')
plt.legend()
plt.title('Loss in train and test')
plt.show()


#

#VISUALIZ E THE FILTERS

#ALEXNET CNN
#VGGNET 
#RESNET

