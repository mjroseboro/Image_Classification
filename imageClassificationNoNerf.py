# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 21:41:41 2024

@author: roseborod
"""

# importing las cosas
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from PIL import ImageOps
import os
# convolutional neural network original architecture
def pad_images_in_folders(input_folder2, output_folder2, target_size):
    
    #ensure the output folder exists
    os.makedirs(output_folder2, exist_ok=True)
    
    #iterate through class folders within the input folder
    for class_folder in os.listdir(input_folder2):
        class_folder_path=os.path.join(input_folder2,class_folder)
        if os.path.isdir(class_folder_path):
            class_output_folder=os.path.join(output_folder2,class_folder)
            os.makedirs(class_output_folder,exist_ok=True)
            
            #iterate through all images in the class folder
            for filename in os.listdir(class_folder_path):
                if filename.endswith (('.jpg', '.png')):
                    input_path=os.path.join(class_folder_path, filename)
                    output_path=os.path.join(class_output_folder,filename)
                    
                    #load and pad the image
                    
                    img=Image.open(input_path)
                    img_padded=ImageOps.fit(img,target_size,method=0, bleed=0.0, centering=(.5,.5))
                    
                    #save padded image to the output folder
                    img_padded.save(output_path)
                    
#example usage for training data
train_input_folder2= 'C:/Users/roseborod/OneDrive - UNC-Wilmington/Documents/input folder training2'
train_output_folder2= 'C:/Users/roseborod/OneDrive - UNC-Wilmington/Documents/output folder training2'
target_size=(244,244)

pad_images_in_folders(train_input_folder2,train_output_folder2,target_size)
    
#example usage for testing data
test_input_folder2= 'C:/Users/roseborod/OneDrive - UNC-Wilmington/Documents/input folder testing2'
test_output_folder2= 'C:/Users/roseborod/OneDrive - UNC-Wilmington/Documents/output folder testing2'
target_size=(244,244)

pad_images_in_folders(test_input_folder2,test_output_folder2,target_size)


#define the model

model2 = models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(244,244,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(1,activation='sigmoid') # binary classification 1 class:ghost gun, 0 class: not ghost gun)
]) # label in data set is 1 or 0, # column 1 is image path to PMF or not PMF folders

#compile baby

model2.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#data preprocessing for training

train_datagen2=ImageDataGenerator(rescale=1./255, shear_range = 0.2, zoom_range=0.2, horizontal_flip=True)

#data preprocessing for testing

test_datagen2=ImageDataGenerator(rescale=1./255)

#load and preprocess the training data
train_generator2=train_datagen2.flow_from_directory('C:/Users/roseborod/OneDrive - UNC-Wilmington/Documents/output folder training2',target_size=(244,244),batch_size=32,class_mode='binary')
test_generator2=test_datagen2.flow_from_directory('C:/Users/roseborod/OneDrive - UNC-Wilmington/Documents/output folder testing2',target_size=(244,244),batch_size=32,class_mode='binary')



#train the model

model2.fit(train_generator2,epochs=10,validation_data=test_generator2)

#save the model
model2.save('ghost_gun_classifier_noNerf.keras')

import sklearn
from sklearn.metrics import confusion_matrix, classification_report

y_pred2=model2.predict(test_generator2)
y_pred2_binary=(y_pred2>.5).astype(int)

cm2= confusion_matrix(test_generator2.classes, y_pred2_binary)
print("Confusion Matrix:")
print(cm2)

report2= classification_report(test_generator2.classes,y_pred2_binary)
print("\nClassification Report:")
print(report2)



from tensorflow.keras.preprocessing import image
import numpy as np

tf.keras.models.load_model('ghost_gun_classifier_noNerf.keras')

#load and pre process input image
new_image_path='C:/Users/roseborod/Downloads/modelimage.jpg'
img=image.load_img(new_image_path, target_size=(244,244))
img_array=image.img_to_array(img)
img_array=np.expand_dims(img_array,axis=0)
img_array/=255.0

new_image_path2='C:/Users/roseborod/Downloads/modelimage2.jpg'
img2=image.load_img(new_image_path2, target_size=(244,244))
img_array2=image.img_to_array(img2)
img_array2=np.expand_dims(img_array2,axis=0)
img_array2/=255.0

new_image_path3='C:/Users/roseborod/Downloads/modelimage3.jpg'
img3=image.load_img(new_image_path3, target_size=(244,244))
img_array3=image.img_to_array(img3)
img_array3=np.expand_dims(img_array3,axis=0)
img_array3/=255.0

#make prediction
prediction=model2.predict(img_array)
prediction2=model2.predict(img_array2)
prediction3=model2.predict(img_array3)

#interpret predictions
threshold=.5
if prediction [0,0]>threshold:
    print("Ghost gun detected!")
else:
    print("No ghost gun detected.")
    
if prediction2 [0,0]>threshold:
    print("Ghost gun detected!")
else:
    print("No ghost gun detected.")   
    
if prediction3 [0,0]>threshold:
    print("Ghost gun detected!")
else:
    print("No ghost gun detected.")  