# WhatsApp Image Classifier

A web application that classifies WhatsApp images into three categories: Document, Generic, and Non-Generic.

## Setup Instructions

1. Make sure you have Python 3.8 or higher installed
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python app.py
   ```
4. Open your web browser and go to: http://localhost:5000

## Project Structure

- `app.py` - Main Flask application
- `requirements.txt` - Python package dependencies
- `keras_model.h5` - Trained model file
- `templates/` - HTML templates
- `static/` - Static files (CSS, images)
- `uploads/` - Temporary folder for uploaded images

## Sample Images

You can use these sample images for testing:
- Document: `static/images/gallery/document1.jpg`
- Non-Generic: `static/images/gallery/non-generic1.jpg`
- Generic: `static/images/gallery/generic1.jpg`

## Troubleshooting

If you encounter any issues:

1. Make sure all files are in the correct locations
2. Check that Python and all dependencies are installed correctly
3. Ensure the model file (`keras_model.h5`) is present in the root directory
4. Check the console for any error messages

## Requirements

- Python 3.8+
- Flask 2.0.1
- TensorFlow 2.12.0
- Pillow 10.0.0
- NumPy 1.23.5
- Werkzeug 2.0.1 