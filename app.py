from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file, send_from_directory
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageOps
import io
import logging
from datetime import datetime
import json
import csv
import zipfile
from io import BytesIO

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

UPLOAD_FOLDER = 'uploads'
HISTORY_FILE = 'classification_history.json'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global model variable
model = None

def load_history():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        return []
    except Exception as e:
        logger.error(f"Error loading history: {str(e)}")
        return []

def save_history(history):
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f)
    except Exception as e:
        logger.error(f"Error saving history: {str(e)}")

def add_to_history(filename, result, probabilities):
    history = load_history()
    history.insert(0, {
        'filename': filename,
        'category': result,
        'probabilities': probabilities,
        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    # Keep only the last 50 classifications
    history = history[:50]
    save_history(history)

def generate_report(classifications):
    output = BytesIO()
    with zipfile.ZipFile(output, 'w') as zipf:
        # Create CSV report
        csv_data = BytesIO()
        writer = csv.writer(csv_data)
        writer.writerow(['Filename', 'Category', 'Document Probability', 'Non-Generic Probability', 'Generic Probability', 'Time'])
        
        for classification in classifications:
            writer.writerow([
                classification['filename'],
                classification['category'],
                classification['probabilities']['Document'],
                classification['probabilities']['Non-Generic'],
                classification['probabilities']['Generic'],
                classification['time']
            ])
        
        csv_data.seek(0)
        zipf.writestr('classification_report.csv', csv_data.getvalue())
    
    output.seek(0)
    return output

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
        test_prediction = model.predict(test_input, verbose=0)
        logger.info(f"Model test prediction shape: {test_prediction.shape}")
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
            logger.info("Image converted to RGB")
        
        # Resize image using the same strategy as in the original classifier
        size = (224, 224)
        image = ImageOps.fit(img, size, Image.Resampling.LANCZOS)  # Using LANCZOS instead of deprecated ANTIALIAS
        logger.info(f"Image resized to {size}")
        
        # Convert to numpy array and normalize exactly as in original
        image_array = np.asarray(image)
        logger.info(f"Image array shape: {image_array.shape}")
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        logger.info(f"Normalized image array shape: {normalized_image_array.shape}")
        
        return normalized_image_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
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
        logger.info("Starting image classification request")
        
        if 'file' not in request.files:
            logger.error("No file part in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("No selected file")
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type. Please upload a JPG, JPEG, or PNG image.'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"Saving file to: {filepath}")
        
        try:
            file.save(filepath)
            logger.info("File saved successfully")
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            return jsonify({'error': 'Error saving uploaded file'}), 500
        
        try:
            # Ensure model is loaded
            if model is None:
                logger.info("Model not loaded, attempting to load...")
                if not load_model():
                    logger.error("Failed to load model")
                    return jsonify({'error': 'Could not load the classification model'}), 500
                logger.info("Model loaded successfully")
            
            # Open and verify image
            try:
                img = Image.open(filepath)
                logger.info(f"Image opened successfully. Mode: {img.mode}, Size: {img.size}")
            except Exception as e:
                logger.error(f"Error opening image: {str(e)}")
                return jsonify({'error': 'Could not open image file'}), 500
            
            # Preprocess image
            try:
                normalized_image = preprocess_image(img)
                data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                data[0] = normalized_image
                logger.info(f"Image preprocessed successfully. Shape: {data.shape}")
            except Exception as e:
                logger.error(f"Error preprocessing image: {str(e)}")
                return jsonify({'error': 'Error processing image'}), 500
            
            # Run prediction
            try:
                prediction = model.predict(data, verbose=0)
                logger.info(f"Prediction completed. Shape: {prediction.shape}, Values: {prediction}")
            except Exception as e:
                logger.error(f"Error during prediction: {str(e)}")
                return jsonify({'error': 'Error during classification'}), 500
            
            # Get probabilities
            doc_prob = float(prediction[0][0])
            non_gen_prob = float(prediction[0][2])
            gen_prob = float(prediction[0][1])
            
            logger.info(f"Probabilities - Document: {doc_prob:.3f}, Non-Generic: {non_gen_prob:.3f}, Generic: {gen_prob:.3f}")
            
            # Determine category
            if doc_prob > 0.80:
                result = "Document"
            elif non_gen_prob > 0.80:
                result = "Non-Generic"
            elif gen_prob > 0.80:
                result = "Generic"
            else:
                result = "Unknown"
            
            logger.info(f"Final classification result: {result}")
            
            # Add to history
            probabilities = {
                'Document': f"{doc_prob:.1%}",
                'Non-Generic': f"{non_gen_prob:.1%}",
                'Generic': f"{gen_prob:.1%}"
            }
            add_to_history(filename, result, probabilities)
            
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
            logger.error(f"Error during classification process: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return jsonify({'error': f'Error during classification: {str(e)}'}), 500
        finally:
            try:
                os.remove(filepath)
                logger.info(f"Temporary file removed: {filepath}")
            except Exception as e:
                logger.error(f"Error removing temporary file: {str(e)}")
            if 'img' in locals():
                img.close()
    
    except Exception as e:
        logger.error(f"Unexpected error in classify_image route: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/history')
def history():
    classifications = load_history()
    return render_template('history.html', classifications=classifications)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        subject = request.form.get('subject')
        message = request.form.get('message')
        
        # Here you would typically send an email or store the message
        # For now, we'll just flash a success message
        flash('Thank you for your message! We will get back to you soon.')
        return redirect(url_for('contact'))
    
    return render_template('contact.html')

@app.route('/download-report')
def download_report():
    try:
        category = request.args.get('category', 'all')
        logger.info(f"Generating report for category: {category}")
        
        classifications = load_history()
        logger.info(f"Total classifications loaded: {len(classifications)}")
        
        if category and category != 'all':
            classifications = [c for c in classifications if c['category'].lower() == category.lower()]
            logger.info(f"Filtered classifications for {category}: {len(classifications)}")
        
        if not classifications:
            logger.warning("No classifications found for report")
            flash('No classifications found for the selected category')
            return redirect(url_for('history'))
        
        # Create a BytesIO object to store the CSV
        output = BytesIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Filename', 'Category', 'Document Probability', 'Non-Generic Probability', 'Generic Probability', 'Time'])
        
        # Write data
        for classification in classifications:
            writer.writerow([
                classification['filename'],
                classification['category'],
                classification['probabilities'].get('Document', '0%'),
                classification['probabilities'].get('Non-Generic', '0%'),
                classification['probabilities'].get('Generic', '0%'),
                classification['time']
            ])
        
        # Create the response
        output.seek(0)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'classification_report_{timestamp}.csv'
        
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name=filename
        )
        
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        flash('Error generating report')
        return redirect(url_for('history'))

@app.route('/filter-history')
def filter_history():
    try:
        category = request.args.get('category', 'all')
        logger.info(f"Filtering history for category: {category}")
        
        classifications = load_history()
        logger.info(f"Total classifications loaded: {len(classifications)}")
        
        if category != 'all':
            classifications = [c for c in classifications if c['category'].lower() == category.lower()]
            logger.info(f"Filtered classifications for {category}: {len(classifications)}")
        
        return render_template('history.html', 
                             classifications=classifications, 
                             selected_category=category)
    except Exception as e:
        logger.error(f"Error filtering history: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        flash('Error loading classification history')
        return redirect(url_for('history'))

@app.route('/samples/<path:filename>')
def serve_sample(filename):
    return send_from_directory('static/samples', filename)

@app.route('/clear-history', methods=['POST'])
def clear_history():
    try:
        # Clear the history file
        with open(HISTORY_FILE, 'w') as f:
            json.dump([], f)
        flash('History cleared successfully')
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        flash('Error clearing history')
    return redirect(url_for('history'))

if __name__ == '__main__':
    # Try to load the model at startup
    if not load_model():
        logger.warning("Failed to load model at startup. Will attempt to load when needed.")
    app.run(debug=True) 