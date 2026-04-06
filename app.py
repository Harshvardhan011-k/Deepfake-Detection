import os

# ✅ Fix TensorFlow threading issues (macOS)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# ==============================
# ✅ EXACT MODEL ARCHITECTURE
# ==============================
def build_model():
    input_shape = (224, 224, 3)
    activation = 'relu'
    padding = 'same'
    droprate = 0.1
    epsilon = 0.001

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.BatchNormalization(input_shape=input_shape))

    model.add(tf.keras.layers.Conv2D(16, 3, activation=activation, padding=padding))
    model.add(tf.keras.layers.MaxPooling2D(2))
    model.add(tf.keras.layers.BatchNormalization(epsilon=epsilon))

    model.add(tf.keras.layers.Conv2D(32, 3, activation=activation, padding=padding))
    model.add(tf.keras.layers.MaxPooling2D(2))
    model.add(tf.keras.layers.BatchNormalization(epsilon=epsilon))
    model.add(tf.keras.layers.Dropout(droprate))

    model.add(tf.keras.layers.Conv2D(64, 3, activation=activation, padding=padding))
    model.add(tf.keras.layers.MaxPooling2D(2))
    model.add(tf.keras.layers.BatchNormalization(epsilon=epsilon))
    model.add(tf.keras.layers.Dropout(droprate))

    model.add(tf.keras.layers.Conv2D(128, 3, activation=activation, padding=padding))
    model.add(tf.keras.layers.MaxPooling2D(2))
    model.add(tf.keras.layers.BatchNormalization(epsilon=epsilon))
    model.add(tf.keras.layers.Dropout(droprate))

    model.add(tf.keras.layers.Conv2D(256, 3, activation=activation, padding=padding))
    model.add(tf.keras.layers.MaxPooling2D(2))
    model.add(tf.keras.layers.BatchNormalization(epsilon=epsilon))
    model.add(tf.keras.layers.Dropout(droprate))

    model.add(tf.keras.layers.Conv2D(512, 3, activation=activation, padding=padding))
    model.add(tf.keras.layers.MaxPooling2D(2))
    model.add(tf.keras.layers.BatchNormalization(epsilon=epsilon))
    model.add(tf.keras.layers.Dropout(droprate))

    model.add(tf.keras.layers.GlobalAveragePooling2D())

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    return model

# ==============================
# ✅ LOAD WEIGHTS
# ==============================
model = build_model()
model.load_weights('models/custom_augmented_model.weights.h5')

# Compile (safe)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ==============================
# CONFIG
# ==============================
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==============================
# PREPROCESSING
# ==============================
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ==============================
# ROUTE
# ==============================
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None
    confidence = None

    if request.method == 'POST':
        file = request.files.get('file')

        if file and file.filename != "":
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            image = preprocess_image(filepath)
            pred = model.predict(image)[0][0]

            print("Prediction value:", pred, flush=True)

            # Prediction + confidence
            if pred > 0.5:
                prediction = "Real"
                confidence = round(pred * 100, 2)
            else:
                prediction = "Fake"
                confidence = round((1 - pred) * 100, 2)

            image_path = filepath

    return render_template(
        'index.html',
        prediction=prediction,
        image_path=image_path,
        confidence=confidence
    )

# ==============================
# RUN
# ==============================
if __name__ == '__main__':
    app.run(debug=False, use_reloader=False)