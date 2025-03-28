from flask import Flask, request, render_template, redirect, url_for
import os
from werkzeug.utils import secure_filename
from src.predict import TumorClassifier
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.secret_key = 'your-secret-key-here'

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

classifier = TumorClassifier()

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('home'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))
    
    if file:
        # Save the uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get prediction
        result = classifier.predict(filepath)
        
        # Clean up if image is out of scope
        if result.get('prediction') in ['Out of Scope', 'Error']:
            os.remove(filepath)
            filename = None
        
        return render_template('index.html', result=result, image_path=filename)
    
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)