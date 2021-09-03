import numpy as np
from keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.optimizer_v2.adam import Adam
from .model import UNet
from .const import *
from .imgutil import load_one_image, data_augmentation
import os
from os import path
import cv2

class CorrosionClassifier:
    def __init__(self, image_size=128):
        self.imgsize=image_size
        self.weight_file='/home/jetbot/pythonAWS/navsea/model/best_UNet_model.h5'
        self.medias = [Type_I, Type_IA, Type_IB, Type_II, Type_III, Type_1, Type_3, Type_5, Type_7, Type_9]
        self.labels = ['image', 'depth map', 'Type_I', 'Type_IA', 'Type_IB', 'Type_II',
                  'Type_III', 'Type_1', 'Type_3', 'Type_5', 'Type_7', 'Type_9']
        print(self.weight_file)
       

        self.model = UNet(input_shape=(image_size, image_size, 3))
        self.model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
        self.load_weightfiles(nweight_file=self.weight_file)
        # Load in pre-trained COCO weights
        #self.model.load_weights(self.weight_file)  # , by_name=True)

    def load_weightfiles(self, nweight_file):
       
       # if path.isfile(nweight_file):
            self.weight_file=nweight_file
            self.model.load_weights(self.weight_file)
            
        #else:
          #  print("model file not found and default model will be used")

    def train(self, imagefolder = 'images', label_folder = 'masks', epochs = 30, lr = 1e-4):
        #beth
        pass

    def savewgtfile(self, wgtfileloc):
        pass

    def train(self):

        image_files = os.listdir('images')
        label_files = os.listdir('masks')

        train_data, train_labels=[],[]

        for image in image_files:
            img = load_one_image('images/'+ str(image))
            train_data.append(img)


        for mask in label_files:
            mask = cv2.imread('masks/'+ str(mask), -1)
            mask = cv2.resize(mask, (self.imgsize, self.imgsize))
            mask = mask.reshape((self.imgsize, self.imgsize))
            mask=mask/255.0
            train_labels.append(mask)

        X_train = np.array(train_data)
        y_train = np.array(train_labels).astype(int)
        y_train = y_train.reshape(len(train_labels), 128, 128, 1)


        ModelCheckpoint('unet_corrosion.hdf5', monitor='loss', verbose=1, save_best_only=True)
        self.model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
        self.model.fit(X_train, y_train, epochs=30, batch_size=16, verbose=1)
        self.model.save('best_UNet_model.h5')

    def predict(self, imfile):
        img= load_one_image(imfile, image_size=self.imgsize)

        data = np.array(img)
        X = data_augmentation(data, label=False, medias=self.medias)
        y_pred = self.model.predict(np.array(X))
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(int)

        # return the orginal picture's augmentation
        y_pred = np.reshape(y_pred[0], (self.imgsize, self.imgsize))
        return y_pred
