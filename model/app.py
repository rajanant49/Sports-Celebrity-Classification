import streamlit as st
from PIL import Image
import cv2
import json
import numpy as np
import pickle
import pywt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("Sports Celebrity Classification")
st.write("Select a image to Classify")

__class_name_to_number = {}
__class_number_to_name = {}
__model = None

def classify_image(image):
    imgs=get_cropped_image_if_2_eyes(image)

    result=[]

    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

        len_image_array=32*32*3+32*32

        final=combined_img.reshape(1,len_image_array).astype(float)

        result.append({
        'class':class_number_to_name(__model.predict(final)[0]),
        'class_probability':np.round(__model.predict_proba(final)*100,2).tolist()[0],
        'class_dictionary': __class_name_to_number
        })
    return result


def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open("./class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}
        
    
    global __model
    pickle_in = open('./saved_model.pkl', 'rb') 
    __model = pickle.load(pickle_in)
    
    print("loading saved artifacts...done")

def get_cropped_image_if_2_eyes(img):
    face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                cropped_faces.append(roi_color)
    return cropped_faces

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversions
    #convert to grayscale
    imArray = cv2.cvtColor( imArray,cv2.COLOR_RGB2GRAY )
    #convert to float
    imArray =  np.float32(imArray)
    imArray /= 255;
    # compute coefficients
    coeffs=pywt.wavedec2(imArray, mode, level=level)

    #Process Coefficients
    coeffs_H=list(coeffs)
    coeffs_H[0] *= 0;

    # reconstruction
    imArray_H=pywt.waverec2(coeffs_H, mode);
    imArray_H *= 255;
    imArray_H =  np.uint8(imArray_H)
    return imArray_H

def load_image(image_file):
    img = Image.open(image_file)
    return img

def get_image():
    image_file = st.file_uploader("Upload Image",type=['png','jpeg','jpg'])
    if image_file is not None:
        file_details = {"Filename":image_file.name,"FileType":image_file.type,"FileSize":image_file.size}
        st.write(file_details)
        img = load_image(image_file)
        img = np.array(img.convert('RGB'))
        st.image(img,width=200,height=200)
        cropped_img=get_cropped_image_if_2_eyes(img)
        if cropped_img is None:
            st.write("Face not Clear .......please provide an image with clear face")
        else:
            st.image(cropped_img,width=200,height=200)
        #st.image(image_array,width=100,height=100)
        return cropped_img

if __name__=='__main__':
    load_saved_artifacts()
    image=get_image()
    print(classify_image(image))


    
