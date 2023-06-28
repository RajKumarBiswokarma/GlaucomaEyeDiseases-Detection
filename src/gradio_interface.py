import gradio as gr
import tensorflow as tf
import cv2
import numpy as np



model = tf.keras.models.load_model("inception_model_epoch.h5")

def preprocess_image(image):
    image = cv2.resize(image,(224,224))
    # transformed_image = cv2.applyColorMap(image, cv2.COLORMAP_HSV)
    # transformed_image = cv2.resize(transformed_image, (224, 224))
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    # ret,thresh = cv2.threshold(gray_image,100,300,0) 
    # contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    # contoured_image = cv2.drawContours(image,contours,-1,(0,300,0),1)
    # contoured_image = cv2.resize(contoured_image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image) / 255
    image = np.expand_dims(image, axis=0)
    return image #, transformed_image, contoured_image

def predict(image):
    # Preprocess the image
    image= preprocess_image(image)

    # Make a prediction using the model
    inception_model_pred = model.predict(image)
    probability = inception_model_pred[0]
    if probability[0] < 0.5:
        result = 'Glaucoma POSITIVE. Consult Ophthalmologist as soon as possible.'
    else:
        result = 'Glaucoma NEGATIVE. You have a Healthy eye.'
        
   
    # Return the predicted class and processed images
    return result




Glaucoma_interface = gr.Interface(predict,
                     inputs=gr.inputs.Image(), 
                     outputs=["text"],
                     title="Glaucoma Prediction Using InceptionResNetV2 architecture (CNN model)",
                     description="Upload an image")

Glaucoma_interface.launch()
