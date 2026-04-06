# Deepfake Detection

This project is a Deepfake Detection web application built with **Flask** and **TensorFlow / Keras**. It allows users to upload an image, processes it through a custom-trained Convolutional Neural Network (CNN), and predicts whether the image is **Real** or **Fake** (Deepfake) along with a confidence percentage.

## Features
- **Modern Web Interface:** A user-friendly UI for uploading images built with HTML/CSS and Flask templates.
- **Deep Learning Model:** Utilizes a custom-built CNN with Batch Normalization, Dropout, and Global Average Pooling to ensure accurate classifications.
- **Real-Time Prediction:** Images are preprocessed and classified instantaneously.
- **Support for Notebooks:** Includes various Jupyter notebooks (`ipynb files/`) showing experiments with different model architectures like DenseNet, VGGFace, and baseline CNNs.

## Tech Stack
- **Backend:** Python, Flask
- **Machine Learning:** TensorFlow, Keras, NumPy, Pillow (PIL)
- **Frontend:** HTML, CSS

## Project Structure
```text
.
├── app.py                # Main Flask application file.
├── models/               # Contains the trained model files (.h5, .keras, .weights.h5).
├── ipynb files/          # Jupyter Notebooks documenting the model training and experimentation process.
├── static/               # CSS styles, images, and uploaded files.
├── templates/            # HTML templates for the web interface.
└── README.md             # Project documentation.
```

## Setup and Installation

### Prerequisites
Make sure you have `python` (3.8+ recommended) and `pip` installed on your system.

### 1. Clone the repository
```bash
git clone https://github.com/Harshvardhan011-k/Deepfake-Detection.git
cd Deepfake-Detection
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies
Ensure you install the required packages. You can install them manually via:
```bash
pip install Flask tensorflow numpy Pillow
```

### 4. Run the Application
Start the Flask server by running:
```bash
python app.py
```

### 5. Open in Browser
Navigate to `http://127.0.0.1:5000/` in your web browser to access the Deepfake Detection interface.

## How it Works
1. The user uploads an image via the web interface.
2. `app.py` receives the image, resizes it to `224x224`, normalizes the pixel values, and prepares it for prediction.
3. The best model evaluates the image and returns a probability score.
4. The prediction (`Fake` or `Real`) and the confidence score are displayed back to the user.
