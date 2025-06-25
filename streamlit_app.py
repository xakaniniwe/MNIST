import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from PIL import Image

st.set_page_config(page_title="🧠 MNIST Streamlit Trainer", layout="centered")
st.title("MNIST Handwritten Digit Classifier (Trained in Streamlit)")
st.write("Upload a 28x28 grayscale digit image (white on black) to classify it.")

# Train the model only once using session state
if "model" not in st.session_state:
    with st.spinner("Training model..."):
        # Load dataset
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Normalize and reshape
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

        # Build model
        model = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=3, activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(64, kernel_size=3, activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(10, activation="softmax")
        ])

        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fit(x_train, y_train, epochs=3, validation_split=0.1, verbose=0)

        st.session_state.model = model
        st.success("Model trained and ready!")

# Upload image
uploaded_file = st.file_uploader("Upload a 28x28 white-on-black PNG", type=["png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("L").resize((28, 28))
    st.image(img, caption="Uploaded Digit", width=150)

    # Preprocess: invert image if white on black
    img_np = 255 - np.array(img)  # invert for white digit on black bg
    img_np = img_np / 255.0
    img_np = img_np.reshape(1, 28, 28, 1).astype("float32")

    # Predict
    prediction = st.session_state.model.predict(img_np)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.success(f"Predicted Digit: {predicted_class} (Confidence: {confidence:.2f})")
