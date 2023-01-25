import os 
import numpy as np

from PIL import Image
import tensorflow as tf
from utilities import LABEL_MAPING,IMAGE_SHAPE,get_label_probabilities, init_images_predicted, init_images_uploaded

#model.summary()
from keras.models import  load_model

def load_np_image(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')/255
    
    new_image = tf.image.resize(np_image, IMAGE_SHAPE)
    new_image = np.expand_dims(new_image, axis=0)
    
    return new_image


def predict(path:str):
    """
    path:
        full path to image
    """
    #load image as array and resize it.
    image = load_np_image(path)
    #make prediction
    answer = model.predict(image)
    #predicted class
    pred = np.argmax(answer[0])

    text = ""
    for i in range(len(LABEL_MAPING)):
        text+= LABEL_MAPING[i] +"--> probability: " +  str( round(answer[0][i],2) ) + "\n"

    print("Predicted class: ",pred )    
    print(text)
    print("Prediction array: ",answer[0])

    return get_label_probabilities(answer[0])

if __name__ == "__main__":
    init_images_uploaded()
    init_images_predicted()

    model = load_model("final_weights.h5")
    image_files= os.listdir('images_uploaded')
    dir = 'images_uploaded'
    for image_file in image_files:
        path =  os.path.join(dir,image_file)
        print("image filename: ",image_file)
        probs = predict(path)
        print(probs)
        print()

