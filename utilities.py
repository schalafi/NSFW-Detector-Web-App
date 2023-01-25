import json
import os 
from typing import Dict 
import numpy as np

from PIL import Image
import tensorflow as tf
#from keras.models import  load_model

IMAGES_PREDICTED_DIR = "images_predicted"
IMAGES_UPLOADED_DIR = "images_uploaded"

#Init image folders
#init_images_predicted()
#init_images_uploaded()


LABEL_MAPING = {0 : "Neutral", 1 : "Porn", 2 : "Sexy"}
IMAGE_SHAPE = [224,224]

def transform_image(file_bytes):
    """
    Process image to pass to our model.

    """
    image = Image.open(file_bytes)
    image = np.array(image).astype('float32')/255
    
    new_image = tf.image.resize(image, IMAGE_SHAPE)
    new_image = np.expand_dims(new_image, axis=0)
    
    return new_image

def get_label_probabilities(array):
    """
    Return a dictionary with 
    {"Neutral":prob,"Porn":prob,"Sexy":prob}
    """
    probs = {}
    for i,l in enumerate(["Neutral","Porn","Sexy"]):
        probs[l] = round(array[i],3)

    return  probs


def predict(image, model = None):
    """
    image:
        numpy array
    """
    #make prediction
    answer = model.predict(image)
    #predicted class
    pred = np.argmax(answer[0])

    text = ""
    for i in range(len(LABEL_MAPING)):
        text+= LABEL_MAPING[i] +" --> probability: " +  str( round(answer[0][i],2) ) + "\n"

    print("Predicted class: ",pred )    
    print(text)
    print("Prediction array: ",answer[0])

    return get_label_probabilities(answer[0])

def only_strings(dictionary: Dict):
    return {str(k): str(float(v)) for k,v in dictionary.items()}

def dict_to_json(dictionary: Dict):
    return json.dumps(
        {str(k): float(v) for k,v in dictionary.items()}
        )

def json_to_dict(json_object):

    return json.loads(json_object)



def init_images_predicted():
    dir_name = "images_predicted"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def init_images_uploaded():
    dir_name = "images_uploaded"
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

