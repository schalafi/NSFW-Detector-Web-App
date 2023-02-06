import os
import io
import numpy as np
from PIL import Image, ImageFilter, ImageDraw,ImageFont
from keras.models import  load_model

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from utilities import (transform_image,predict, dict_to_json, 
    json_to_dict,only_strings,init_images_predicted,init_images_uploaded,
    IMAGES_PREDICTED_DIR, IMAGES_UPLOADED_DIR)


#RUN: uvicorn service:app --reload 
model = load_model("final_weights.h5")

#Assign an instance of the FastAPI class to the variable "app".
# You will interact with your api using this instance.
app = FastAPI(title='NSFW Detection with DL.')

# By using @app.get("/") you are allowing the GET method to work for the / endpoint.
@app.get("/")
def home():
    return "Congratulations! Your API is working as expected. Now head over to http://localhost:8000/docs."

# This endpoint handles all the logic necessary for the object detection to work.
# It requires the desired model and the image in which to perform object detection.
@app.post("/predict") 
async def prediction(confidence:float = 0.2,file: UploadFile = File(...)):

    assert  0 <= confidence  and confidence <=1, f"confidence must be a float >0 and <= 1, given {confidence}"

    # 1. VALIDATE INPUT FILE
    filename = file.filename
    fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")
    
    # 2. TRANSFORM RAW IMAGE INTO array
    file_bytes = file.file

    # Read image from bytes
    # normalize values and resize to (1,224,224,3)
    image = transform_image(file_bytes)
    
    # 3. RUN NSFW DETECTION MODEL
    preds = predict(image,model)

    labels = []
    #ADD labels
    if preds.get('Sexy', None) > 0.15: labels.append('sexy')
    if preds.get('Porn',None) >= confidence: labels.append('porn')
    
    #read original image
    orig_image = Image.open(file_bytes)
    print("Orig image type: ", type(orig_image))

    #Blur image if porn is detected
    if 'porn' in labels:
        orig_image =orig_image.filter(ImageFilter.GaussianBlur(25))
    
    #print("FONTS:",os.system("locate .ttf"))

    if 'sexy' in labels:
        # draw image object
        #my_font = ImageFont.truetype("arial.ttf", 85)
        #Add text label to image "sexy"
        w,h = orig_image.size
        font_size = int(float(w) / 3)
        font_path ="/"+os.path.join(os.path.dirname(__file__)[3:-1],'magical_story/' + "Magical Story.ttf")
        print("FONT PATH: ",font_path + '\n', font_size)
        print("Os file exists: ", os.path.exists(font_path) )
        my_font = ImageFont.truetype(font_path, font_size)
        edit   = ImageDraw.Draw(orig_image)

        # add text to image
        edit.text((10, 10),"SEXY",
            font = my_font,
            fill=(255, 0, 0))

    name, ext = os.path.splitext(filename)

    # Save it in a folder within the server
    new_name = get_secure_filename(name).replace(" ", "")
    file_location =os.path.join(IMAGES_PREDICTED_DIR, new_name  +  ext)
    print("Orig filename: ", filename)
    print("Secure path for image:", file_location)
    
    #TODO: save image if necessary in uploaded_images
    #get bytes object
    bytes_image = image_to_byte_array(orig_image)
    #convert predictions numbers to strings
    dict_without_np_float = only_strings(preds)

    return Response(
        content = bytes_image,
        #send array of predictions in headers
        # a dict of {category: probability}
        headers =dict_without_np_float,
        media_type="image/jpeg" )

def image_to_byte_array(image: Image) -> bytes:
    # BytesIO is a fake file stored in memory
    img_byte_array = io.BytesIO()
    print("IMAGE FORMAT: ", image.format)
    # image.save expects a file as a argument, passing a bytes io instance
    #format=image.format
    image.save(img_byte_array, format='jpeg')
    # Turn the BytesIO object back into a bytes object
    bytes_object = img_byte_array.getvalue()
    return bytes_object

def get_secure_filename(filename: str) -> str:
    fn = "".join([c for c in filename if c.isalpha() or c.isdigit() or c==' ']).rstrip()
    return fn  

#REFERENCES
#https://stackoverflow.com/questions/24219446/render-image-without-saving
#https://stackoverflow.com/questions/33101935/convert-pil-image-to-byte-array
#https://stackoverflow.com/questions/32908639/open-pil-image-from-byte-file