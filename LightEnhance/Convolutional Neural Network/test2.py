import numpy as np
import pandas as pd
import os
#clear terminal
os.system('cls' if os.name == 'nt' else 'clear')
os.environ["TF_ENABLE_ONEDNN_OPTS"]="FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2 as cv
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input,decode_predictions
from keras import backend as K
from keras.layers import add, Conv2D,MaxPooling2D,UpSampling2D,Input,BatchNormalization, RepeatVector, Reshape
from keras.layers import concatenate
from keras.models import Model
import tensorflow

tensorflow.random.set_seed(2)
np.random.seed(1)

def noisy(noise_typ,image):
    if noise_typ == "gauss":
        row,col,ch= image.shape
        mean = 0
        var = 0.0001
        sigma = var**0.05
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row, col, ch) if ch > 1 else gauss.reshape(row, col)

        noisy =  gauss + image
        return noisy
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.04  # Adjust this value if needed
        out = np.copy(image)

    # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p).astype(int)
        salt_coords = [np.random.randint(0, i, num_salt) for i in image.shape[:2]]
        out[salt_coords[0], salt_coords[1], :] = 1  # Salt mode

    # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p)).astype(int)
        pepper_coords = [np.random.randint(0, i, num_pepper) for i in image.shape[:2]]
        out[pepper_coords[0], pepper_coords[1], :] = 0  # Pepper mode

        return out



    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ =="speckle":
        row,col,ch = image.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row, col, ch) if ch > 1 else gauss.reshape(row, col)
       
        noisy = image + image * gauss
        return noisy

def PreProcessData(ImagePath):
    X_=[]
    y_=[]
    count=0
    for imageDir in os.listdir(ImagePath):
        if count<2131:
            try:
                count=count+1
                img = cv.imread(ImagePath + imageDir)
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img_y = cv.resize(img,(500,500))
                hsv = cv.cvtColor(img_y, cv.COLOR_BGR2HSV) #convert it to hsv
                hsv[...,2] = hsv[...,2]*0.2
                img_1 = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
                Noisey_img = noisy("s&p",img_1)
                X_.append(Noisey_img)
                y_.append(img_y)
            except:
                pass
    X_ = np.array(X_)
    y_ = np.array(y_)
    
    return X_,y_

def InstantiateModel(in_):
    
    model_1 = Conv2D(16,(3,3), activation='relu',padding='same',strides=1)(in_)
    model_1 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model_1)
    model_1 = Conv2D(64,(2,2), activation='relu',padding='same',strides=1)(model_1)
    
    model_2 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(in_)
    model_2 = Conv2D(64,(2,2), activation='relu',padding='same',strides=1)(model_2)
    
    model_2_0 = Conv2D(64,(2,2), activation='relu',padding='same',strides=1)(model_2)
    
    model_add = add([model_1,model_2,model_2_0])
    
    model_3 = Conv2D(64,(3,3), activation='relu',padding='same',strides=1)(model_add)
    model_3 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model_3)
    model_3 = Conv2D(16,(2,2), activation='relu',padding='same',strides=1)(model_3)
    
    model_3_1 = Conv2D(32,(3,3), activation='relu',padding='same',strides=1)(model_add)
    model_3_1 = Conv2D(16,(2,2), activation='relu',padding='same',strides=1)(model_3_1)
    
    model_3_2 = Conv2D(16,(2,2), activation='relu',padding='same',strides=1)(model_add)
    
    model_add_2 = add([model_3_1,model_3_2,model_3])
    
    model_4 = Conv2D(16,(3,3), activation='relu',padding='same',strides=1)(model_add_2)
    model_4_1 = Conv2D(16,(3,3), activation='relu',padding='same',strides=1)(model_add)
    #Extension
    model_add_3 = add([model_4_1,model_add_2,model_4])
    
    model_5 = Conv2D(16,(3,3), activation='relu',padding='same',strides=1)(model_add_3)
    model_5 = Conv2D(16,(2,2), activation='relu',padding='same',strides=1)(model_add_3)
    
    model_5 = Conv2D(3,(3,3), activation='relu',padding='same',strides=1)(model_5)
    
    return model_5

def GenerateInputs(X,y):
    for i in range(len(X)):
        X_input = X[i].reshape(1,500,500,3)
        y_input = y[i].reshape(1,500,500,3)
        yield (X_input,y_input)

def ExtractTestInput(ImagePath):
    img = cv.imread(ImagePath)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img_ = cv.resize(img,(500,500))
    hsv = cv.cvtColor(img_, cv.COLOR_BGR2HSV) #convert it to hsv
    hsv[...,2] = hsv[...,2]*0.2
    img1 = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    Noise = noisy("s&p",img1)
    Noise = Noise.reshape(1,500,500,3)
    return Noise

def get_enhanced_image(ImagePath):
    image_for_test = ExtractTestInput(ImagePath)
    Prediction = Model_Enhancer.predict(image_for_test)

    Image_test = ImagePath
    img_ = cv.imread(Image_test)
    img_ = cv.resize(img_, (500, 500))

    img_ = img_
    img_ = img_.reshape(500,500,3)
    img_ = cv.cvtColor(img_, cv.COLOR_BGR2RGB)
    img_ = img_.reshape(500,500,3)

    img_[:,:,:] = Prediction[:,:,:]
    img_ = cv.cvtColor(img_, cv.COLOR_BGR2RGB)
    img_ = img_.reshape(500,500,3)
    return img_

InputPath="wm-nowm/"
X_,y_ = PreProcessData(InputPath)

K.clear_session()

Input_Sample = Input(shape=(500, 500,3))
Output_ = InstantiateModel(Input_Sample)
Model_Enhancer = Model(inputs=Input_Sample, outputs=Output_)
Model_Enhancer.compile(optimizer="adam", loss='mean_squared_error')
Model_Enhancer.summary()

user_input = "n" 
if user_input == "y":
    Model_Enhancer.fit(GenerateInputs(X_,y_),epochs=10,verbose=1,steps_per_epoch=39,shuffle=True)
    Model_Enhancer.save("Enhancer.h5")
else:
    Model_Enhancer.load_weights("Enhancer.h5")

""" test the model
TestPath="wm-nowm/"
ImagePath=TestPath+"adler-bird-bird-of-prey-raptor-53587.jpeg"
image_for_test = ExtractTestInput(ImagePath)
Prediction = Model_Enhancer.predict(image_for_test)

Image_test=TestPath+"adler-bird-bird-of-prey-raptor-53587.jpeg"

img_1 = cv.imread(Image_test)
img_1 = cv.resize(img_1, (500, 500))
cv.imshow("Ground Truth",img_1)

img_ = ExtractTestInput(Image_test)
#change color of image to RGB
img_ = img_.reshape(500,500,3)
img_ = cv.cvtColor(img_, cv.COLOR_BGR2RGB)
img_ = img_.reshape(500,500,3)
cv.imshow("Low Light Image",img_)

img_[:,:,:] = Prediction[:,:,:]
img_ = cv.cvtColor(img_, cv.COLOR_BGR2RGB)
img_ = img_.reshape(500,500,3)
cv.imshow("Enhanced Image",img_)
cv.waitKey(0)
"""

test_image = "ExDark/Car/2015_02412.jpg"
cv.imshow("Original Image",cv.imread(test_image))
cv.imshow("Enhanced Image",get_enhanced_image(test_image))
cv.waitKey(0)




