from flask import Flask, request, render_template, redirect, url_for, send_file, send_from_directory
import os
import pandas as pd
from werkzeug.utils import secure_filename
from datetime import datetime
import io
import sys

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import TumorClassifier

app = Flask(__name__)

# Create an 'uploads' directory in the project root instead of static/uploads
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.secret_key = 'your-secret-key-here'

# Ensure upload folder exists
upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), UPLOAD_FOLDER)
os.makedirs(upload_dir, exist_ok=True)

print(f"Upload directory: {upload_dir}")  # Debug info

classifier = TumorClassifier()

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files directly"""
    return send_from_directory(upload_dir, filename)

@app.route('/predict', methods=['POST'])
def predict():
    if 'files[]' not in request.files:
        return redirect(url_for('home'))
    
    files = request.files.getlist('files[]')
    if not files or files[0].filename == '':
        return redirect(url_for('home'))
    
    results = []
    timestamp_batch = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for file in files:
        if file:
            # Save the uploaded file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}_{secure_filename(file.filename)}"
            filepath = os.path.join(upload_dir, filename)
            file.save(filepath)
            
            # Get prediction
            result = classifier.predict(filepath)
            
            # Format result for display and Excel
            prediction = result.get('prediction', 'Error')
            
            # Determine if it's a tumor or not
            tumor_status = "No Tumor"
            if prediction not in ['Out of Scope', 'Error']:
                # Assuming your classifier returns a class name or probability
                if prediction == "Tumor" or prediction.lower() == "positive":
                    tumor_status = "Tumor"
            
            # Create the image URL for the template
            image_url = None
            if prediction not in ['Out of Scope', 'Error']:
                image_url = filename
            else:
                # Clean up if image is out of scope
                os.remove(filepath)
            
            # Save result
            results.append({
                'filename': file.filename,
                'image_url': image_url,
                'tumor_status': tumor_status,
                'original_prediction': prediction
            })
    
    # Create Excel file
    if results:
        # Create the server URL based on the request
        server_url = request.url_root.rstrip('/')
        
        excel_data = pd.DataFrame([
            {
                'Image Name': item['filename'],
                'Image Link': f"{server_url}/uploads/{item['image_url']}" if item['image_url'] else "N/A",
                'Tumor Status': item['tumor_status']
            } for item in results
        ])
        
        # Save Excel to disk
        excel_filename = f"tumor_results_{timestamp_batch}.xlsx"
        excel_filepath = os.path.join(upload_dir, excel_filename)
        excel_data.to_excel(excel_filepath, index=False, engine='openpyxl')
        
        print(f"Excel file saved at: {excel_filepath}")  # Debug info
        
        return render_template('index.html', results=results, excel_available=True, excel_filename=excel_filename)
    
    return redirect(url_for('home'))

@app.route('/download/<filename>')
def download_file(filename):
    # Use the absolute path for the file
    file_path = os.path.join(upload_dir, filename)
    print(f"Attempting to download file from: {file_path}")  # Debug info
    
    # Check if file exists before attempting to send
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return "File not found", 404
        
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    print(f"Starting Flask app in debug mode")
    app.run(debug=True)