# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 14:48:27 2019

@author: jkunka
"""

import os
import pandas as pd
import shutil
from scipy.io import loadmat
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
from keras.layers import Dense,Dropout,Flatten
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.callbacks import History,ModelCheckpoint

mat = loadmat('imagelabels.mat')
matdata = mat['labels']
data = np.array(matdata)
source = r"C:\Users\jkunka\Desktop\Tech challenge\transfer learning\jpg"+"\\"
file_names = [fn for fn in os.listdir(r"C:\Users\jkunka\Desktop\Tech challenge\transfer learning\jpg") if fn.endswith('jpg')]

df = pd.DataFrame({'labels':data[0,:],'images':file_names})

print(df.head())

labels_lis = list(df['labels'].unique())
for label in labels_lis:
    if not os.path.exists(os.path.join(r"C:\Users\jkunka\Desktop\Tech challenge\transfer learning",str(label))):
        os.mkdir(os.path.join(r"C:\Users\jkunka\Desktop\Tech challenge\transfer learning",str(label)))
    for i in range(len(df)):
        if df['labels'][i] == label:
            shutil.copy(source+str(df['images'][i]),os.path.join(r"C:\Users\jkunka\Desktop\Tech challenge\transfer learning",str(label)))

HEIGHT = 300
WIDTH = 300
TRAIN_DIR = "Train"
batch_size = 20
Num_Epocs = 10
num_training_images = 8189

base_model = ResNet50(include_top=False,weights='imagenet',input_shape=(HEIGHT,WIDTH,3))
train_data_gen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                    rotation_range=90,horizontal_flip=True,vertical_flip=True)
train_gen = train_data_gen.flow_from_directory(TRAIN_DIR,target_size=(HEIGHT,WIDTH),batch_size=batch_size)

def build_model(base_model,dropout,fc_layers,num_classes):
    for layer in base_model.layers:
        layer.trainable = False
    
    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        x = Dense(fc,activation='relu')(x)
        x = Dropout(dropout)(x)
    
    predictions = Dense(num_classes,activation='softmax')(x)
    
    build_model = Model(inputs=base_model.input,outputs=predictions)
    
    return build_model

class_len = len(labels_lis)
dropout = 0.5
fc_layers = [1024,1024]

built_model = build_model(base_model,dropout=dropout,fc_layers=fc_layers,num_classes=class_len)

adam  = Adam(lr=0.001)

built_model.compile(adam,loss='categorical_crossentropy',metrics=['accuracy'])
filepath="./checkpoints/" + "ResNet50" + "_model_weights.h5"
checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
callbacks_list = [checkpoint]

history = built_model.fit_generator(train_gen, epochs=Num_Epocs, workers=8, 
                                       steps_per_epoch=num_training_images // batch_size, 
                                       shuffle=True, callbacks=callbacks_list)

def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    # plt.figure()
    # plt.plot(epochs, loss, 'r.')
    # plt.plot(epochs, val_loss, 'r-')
    # plt.title('Training and validation loss')
    plt.show()

    plt.savefig('acc_vs_epochs.png')

plot_training(history)

# Plot the training and validation loss + accuracy


#print(len(file_names))

#df = pd.read_excel('labels.xlsx')
#
#for i in range(len(df)):
#    df['image'][i] = 