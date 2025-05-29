import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import os
from os.path import isfile, join
import shutil
import imghdr

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

def classify(image_path):
    # Load the model
    model = tensorflow.keras.models.load_model('keras_model.h5')
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    try:
        img = Image.open(image_path)
        size = (224, 224)

        # resize the image to a 224x224 with the same strategy as in TM2:
        # resizing the image to be at least 224x224 and then cropping from the center
        image = ImageOps.fit(img, size, Image.ANTIALIAS)

        # turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # run the inference
        prediction = model.predict(data).tolist()
        doc = prediction[0][0]
        non_gen = prediction[0][2]
        gen = prediction[0][1]

        if doc > 0.80:
            return "Document"
        elif non_gen > 0.80:
            return "Non-Generic"
        elif gen > 0.80:
            return "Generic"
        else:
            return "Unknown"

    except IOError:
        return "Error processing image"
    finally:
        img.close() 