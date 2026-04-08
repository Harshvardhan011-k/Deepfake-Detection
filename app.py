import os

# ✅ Fix TensorFlow threading issues (important for deployment)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Deepfake Detector AI", layout="centered")

st.title("🧠 Deepfake Detector AI")
st.write("Upload an image to verify its authenticity using deep learning.")

# ==============================
# MODEL ARCHITECTURE
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
# LOAD MODEL (CACHED)
# ==============================
@st.cache_resource
def load_model():
    model = build_model()
    model.load_weights('models/custom_augmented_model.weights.h5')
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = load_model()

# ==============================
# PREPROCESSING
# ==============================
def preprocess_image(image):
    img = image.convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ==============================
# FILE UPLOAD
# ==============================
uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 Analyze Image"):
        with st.spinner("Processing..."):
            img = preprocess_image(image)
            pred = model.predict(img)[0][0]

            # Prediction logic
            if pred > 0.5:
                prediction = "Real"
                confidence = pred * 100
                st.success(f"✅ {prediction}")
            else:
                prediction = "Fake"
                confidence = (1 - pred) * 100
                st.error(f"⚠️ {prediction}")

            # Confidence display
            st.write(f"**Confidence: {confidence:.2f}%**")
            st.progress(int(confidence))
