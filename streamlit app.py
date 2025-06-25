import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="🧠 MNIST Digit Classifier", layout="centered")
st.title("MNIST Handwritten Digit Classifier")
st.write("Upload a 28×28 grayscale digit image to classify it immediately.")

@st.cache_resource
def build_and_train_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28,28,1)),
        tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128,activation='relu'),
        tf.keras.layers.Dense(10,activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Train on MNIST directly in-app:
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1,28,28,1).astype('float32')/255.0
    model.fit(x_train, y_train, epochs=3, batch_size=128, verbose=0)
    return model

model = build_and_train_model()

file = st.file_uploader("Upload your 28×28 digit PNG", type=["png","jpg","jpeg"])
if file:
    img = Image.open(file).convert("L").resize((28,28))
    st.image(img, width=150, caption="Your image")
    arr = np.array(img).reshape(1,28,28,1)/255.0
    pred = model.predict(arr)
    digit = int(np.argmax(pred))
    st.success(f"📌 Predicted Digit: **{digit}** (confidence: {np.max(pred):.2%})")
