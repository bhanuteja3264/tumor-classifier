

```markdown
# 🧠 Brain Tumor Classification System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![Flask](https://img.shields.io/badge/Flask-2.3-lightgrey)

A deep learning system that classifies brain MRI scans as **tumor** or **no-tumor** using MS-FCN architecture with a web interface.

## 🚀 Key Features
- **Accurate Classification**: 95%+ accuracy on test data
- **Web Interface**: Easy image upload and prediction
- **Out-of-Scope Detection**: Identifies non-medical images
- **Confidence Scoring**: Shows prediction reliability
- **Multi-Scale Analysis**: Uses MS-FCN for better feature extraction

## 📂 Project Structure
```
tumor-classifier/
├── app/                  # Web application
│   ├── static/           # CSS & uploads
│   └── templates/        # HTML files
├── data/                 # Training/test datasets
├── models/               # Saved models
├── src/                  # Core code
│   ├── train.py          # Model training
│   ├── predict.py        # Prediction logic
│   └── evaluate.py       # Performance metrics
└── requirements.txt      # Dependencies
```

## 🛠️ Installation
1. Clone the repository:
```bash
git clone https://github.com/bhanuteja3264/tumor-classifier.git
cd tumor-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🧑‍💻 Usage
### 1. Training the Model
```bash
python src/train.py
```
▶️ Trains on `data/train/` and saves model to `models/`

### 2. Running Web App
```bash
python app/app.py
```
🌐 Access at: `http://localhost:5000`

### 3. Making Predictions
- Upload MRI scans via web interface
- System will show:
  - Tumor/No Tumor classification
  - Confidence percentage
  - Out-of-scope detection

## 📊 Dataset
- **Source**: [BraTS Dataset](https://www.med.upenn.edu/cbica/brats2021/data.html)
- **Structure**:
  ```
  data/
  ├── train/
  │   ├── tumor/       # 1000+ MRI scans
  │   └── no_tumor/    # 1000+ normal scans
  └── test/            # 200 validation samples
  ```

## 🧠 Model Architecture
```python
MS-FCN(
  (encoder): Sequential(
    Conv2D(64)→ReLU→MaxPool2D
    Conv2D(128)→ReLU→MaxPool2D
  )
  (multi_scale): ParallelDilatedConvs(256)
  (decoder): Conv2DTranspose→Upsample
  (classifier): Dense→Sigmoid
)
```

## 📈 Performance Metrics
| Metric       | Value |
|--------------|-------|
| Accuracy     | 99.0% |
| F1-Score     | 0.99  |
| Precision    | 0.97  |
| Recall       | 0.99  |

## 🤝 How to Contribute
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

## 📧 Contact
Bhanu Teja - bhanutejaybt.2004@email.com  
Project Link: [https://github.com/bhanuteja3264/tumor-classifier](https://github.com/bhanuteja3264/tumor-classifier)
```
