import gradio as gr
import tensorflow as tf
import cv2
import numpy as np



model = tf.keras.models.load_model("inception_model_epoch.h5")

def preprocess_image(image):
      """Preprocesses an input image for further analysis or model inference.

    Args:
        image (numpy.ndarray): The input image as a NumPy array in BGR format.

    Returns:
        numpy.ndarray: The preprocessed image as a NumPy array with the following transformations:
            - Resized to a shape of (224, 224)
            - Converted from BGR to RGB color space
            - Normalized by dividing pixel values by 255
            - Expanded dimensions to have a shape of (1, 224, 224, 3)

    """
    image = cv2.resize(image,(224,224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image) / 255
    image = np.expand_dims(image, axis=0)
    return image #, transformed_image, contoured_image

def predict(image):
    
    """Performs glaucoma prediction on the input image using a pre-trained model.

    Args:
        image (numpy.ndarray): The input image as a NumPy array in BGR format.

    Returns:
        str: The predicted result indicating the presence or absence of glaucoma:
            - If glaucoma is predicted (probability < 0.5):
                "Glaucoma POSITIVE. Consult Ophthalmologist as soon as possible."
            - If glaucoma is not predicted (probability >= 0.5):
                "Glaucoma NEGATIVE. You have a Healthy eye."

    """
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
