import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2 as cv
import matplotlib.pyplot as plt
import random
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input,decode_predictions
from keras import backend as K
from keras.layers import add, Conv2D,MaxPooling2D,UpSampling2D,Input,BatchNormalization, RepeatVector, Reshape
from keras.layers import concatenate
from keras.models import Model
#from keras.preprocessing.image import ImageDataGenerator




#clear the terminal
os.system('cls' if os.name == 'nt' else 'clear')

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
    
def get_random_wmnowm_image_path():
    base_path = "wm-nowm"
    files = [f for f in os.listdir(base_path) if os.path.isfile(os.path.join(base_path, f))]
    random_image = random.choice(files)
    image_path = os.path.join(base_path, random_image)
    return image_path

def PreProcessData(DirPath):
    X_=[]
    y_=[]
    count=0
    for imageName in os.listdir(DirPath):
        if count<2131:
            try:
                count=count+1
                img = cv.imread(DirPath + imageName)
                #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                img = cv.resize(img,(500,500))
                hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) #convert it to hsv
                hsv[...,2] = hsv[...,2]*0.2
                img_1 = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
                Noisey_img = noisy("s&p",img_1)
                X_.append(Noisey_img)
                y_.append(img)
            except:
                pass
    #X_ = np.array(X_)
    #y_ = np.array(y_)
    
    return X_,y_

#image_path = get_random_wmnowm_image_path()

X,y = PreProcessData("wm-nowm/")

K.clear_session()
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

Input_Sample = Input(shape=(500, 500,3))
Output_ = InstantiateModel(Input_Sample)
Model_Enhancer = Model(inputs=Input_Sample, outputs=Output_)
Model_Enhancer.compile(optimizer="adam", loss='mean_squared_error')
Model_Enhancer.summary()


def GenerateInputs(X,y):
    for i in range(len(X)):
        X_input = X[i].reshape(1,500,500,3)
        y_input = y[i].reshape(1,500,500,3)
        yield (X_input,y_input)
#Model_Enhancer.fit(GenerateInputs(X,y),epochs=53,verbose=1,steps_per_epoch=39,shuffle=True)
Model_Enhancer.fit(GenerateInputs(X,y),epochs=5,verbose=1,steps_per_epoch=10,shuffle=True)
TestPath="wm-nowm/"

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

ImagePath=TestPath+"bench-man-people-woman.jpg"
image_for_test = ExtractTestInput(ImagePath)
Prediction = Model_Enhancer.predict(image_for_test)
Model_Enhancer.save("Model_Enhancer.h5")
#how do i import the saved model
K.models.load_model("Model_Enhancer.h5")

Prediction = Prediction.reshape(500,500,3)
plt.imshow(Prediction)
plt.show()

Image_test=TestPath+"bench-man-people-woman.jpg"
plt.figure(figsize=(30,30))
plt.subplot(5,5,1)
img_1 = cv.imread(Image_test)
img_1 = cv.cvtColor(img_1, cv.COLOR_BGR2RGB)
img_1 = cv.resize(img_1, (500, 500))
plt.title("Ground Truth",fontsize=20)
plt.imshow(img_1)


plt.subplot(5,5,1+1)
img_ = ExtractTestInput(Image_test)
img_ = img_.reshape(500,500,3)
plt.title("Low Light Image",fontsize=20)
plt.imshow(img_)


plt.subplot(5,5,1+2)
img_[:,:,:] = Prediction[:,:,:]
plt.title("Enhanced Image",fontsize=20)
plt.imshow(img_)


TestPath2 = r"C:/Users/Morteza/OneDrive/Desktop/YouTube/coding/LLIE/travel and adventure/"

Image_test2 = TestPath2 + "10.jpg"
plt.figure(figsize=(30,30))
plt.subplot(5,5,1)
img_1 = cv.imread(Image_test2)
img_1 = cv.cvtColor(img_1, cv.COLOR_BGR2RGB)
img_1 = cv.resize(img_1, (500, 500))
plt.title("Ground Truth",fontsize=20)
plt.imshow(img_1)


plt.subplot(5,5,1+1)
img_ = ExtractTestInput(Image_test2)
Prediction = Model_Enhancer.predict(img_)
img_ = img_.reshape(500,500,3)
plt.title("Low Light Image",fontsize=20)
plt.imshow(img_)


plt.subplot(5,5,1+2)
Prediction = Prediction.reshape(500,500,3)
img_[:,:,:] = Prediction[:,:,:]
plt.title("Enhanced Image",fontsize=20)
plt.imshow(img_)



