import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load the trained model
model = tf.keras.models.load_model("mnist_tf2_10_fixed_model.h5")


st.title("MNIST Digit Classifier 🧠")
st.write("Upload a 28x28 pixel image of a handwritten digit.")

uploaded_file = st.file_uploader("Choose a digit image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert for MNIST format
    image = image.resize((28, 28))  # Resize
    st.image(image, caption="Uploaded Image", use_column_width=False)

    img_array = np.array(image).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array)
    pred_class = np.argmax(prediction)

    st.subheader(f"Predicted Digit: {pred_class}")
