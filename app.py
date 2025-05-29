from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageOps
import io
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}  # Added PNG support

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global model variable
model = None

def load_model():
    global model
    try:
        model_path = 'keras_model.h5'
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return False
        
        logger.info("Loading model...")
        model = keras.models.load_model(model_path)
        # Test the model with a simple prediction to ensure it's working
        test_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
        model.predict(test_input)
        logger.info("Model loaded and tested successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img):
    try:
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize image
        size = (224, 224)
        img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        image_array = np.asarray(img)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        
        return normalized_image_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

def classify(filepath):
    global model
    try:
        logger.info(f"Starting classification for file: {filepath}")
        
        if not os.path.exists(filepath):
            logger.error(f"File not found at {filepath}")
            return "Error: File not found"
        
        if model is None:
            logger.info("Model not loaded, attempting to load...")
            if not load_model():
                return "Error: Could not load model"
        
        # Load and preprocess image
        try:
            img = Image.open(filepath)
            logger.info(f"Image opened successfully. Mode: {img.mode}, Size: {img.size}")
        except Exception as e:
            logger.error(f"Error opening image: {str(e)}")
            return "Error: Could not open image"
        
        try:
            normalized_image = preprocess_image(img)
            data = np.expand_dims(normalized_image, axis=0)
            logger.info("Image preprocessed successfully")
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return "Error: Could not process image"
        
        try:
            prediction = model.predict(data, verbose=0)
            logger.info(f"Raw prediction: {prediction}")
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return "Error: Classification failed"
        
        # Get prediction probabilities
        doc = float(prediction[0][0])
        non_gen = float(prediction[0][2])
        gen = float(prediction[0][1])
        
        logger.info(f"Probabilities - Document: {doc:.3f}, Non-Generic: {non_gen:.3f}, Generic: {gen:.3f}")
        
        # Return the category with highest probability
        max_prob = max(doc, non_gen, gen)
        if max_prob < 0.5:  # Lower threshold for better results
            return "Unknown"
        
        if doc == max_prob:
            return "Document"
        elif non_gen == max_prob:
            return "Non-Generic"
        else:
            return "Generic"
            
    except Exception as e:
        logger.error(f"Unexpected error during classification: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return "Error during classification"
    finally:
        if 'img' in locals():
            img.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            result = classify(filepath)
            flash(f'File successfully uploaded and classified as: {result}')
            return redirect(url_for('index'))
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    try:
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            logger.info(f"Saving file to: {filepath}")
            file.save(filepath)
            
            try:
                # Get the raw prediction probabilities
                img = Image.open(filepath)
                normalized_image = preprocess_image(img)
                data = np.expand_dims(normalized_image, axis=0)
                prediction = model.predict(data, verbose=0)
                
                # Get individual probabilities
                doc_prob = float(prediction[0][0])
                non_gen_prob = float(prediction[0][2])
                gen_prob = float(prediction[0][1])
                
                # Determine the result category
                max_prob = max(doc_prob, non_gen_prob, gen_prob)
                if max_prob < 0.5:
                    result = "Unknown"
                elif doc_prob == max_prob:
                    result = "Document"
                elif non_gen_prob == max_prob:
                    result = "Non-Generic"
                else:
                    result = "Generic"
                
                logger.info(f"Classification result: {result}")
                logger.info(f"Probabilities - Document: {doc_prob:.3f}, Non-Generic: {non_gen_prob:.3f}, Generic: {gen_prob:.3f}")
                
                response_data = {
                    'success': True,
                    'result': result,
                    'probabilities': {
                        'document': doc_prob,
                        'non_generic': non_gen_prob,
                        'generic': gen_prob
                    },
                    'filename': filename
                }
                
                return jsonify(response_data)
                
            except Exception as e:
                logger.error(f"Error during classification: {str(e)}")
                return jsonify({'error': 'Error during classification'}), 500
            finally:
                try:
                    os.remove(filepath)
                    logger.info(f"Temporary file removed: {filepath}")
                except Exception as e:
                    logger.error(f"Error removing temporary file: {str(e)}")
                if 'img' in locals():
                    img.close()
        
        logger.error("Invalid file type")
        return jsonify({'error': 'Invalid file type. Please upload a JPG, JPEG, or PNG image.'}), 400
        
    except Exception as e:
        logger.error(f"Error in classify_image route: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

if __name__ == '__main__':
    # Try to load the model at startup
    if not load_model():
        logger.warning("Failed to load model at startup. Will attempt to load when needed.")
    app.run(debug=True) 