import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the trained model
model = load_model("pneumonia_cnn_model.keras")

# Set title
st.title("Pneumonia Detection from Chest X-Ray 🩻")

# File uploader for image
uploaded_file = st.file_uploader("Upload a Chest X-ray Image", type=["jpg", "jpeg", "png"])

# Predict function
def predict_pneumonia(img_path):
    img = image.load_img(img_path, target_size=(320, 320))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0   # rescale
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    
    if prediction >= 0.5:
        result = "Pneumonia Detected 😷"
    else:
        result = "Normal Chest X-Ray ✅"
    return result, prediction

# If an image is uploaded
if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded X-Ray Image', use_column_width=True)

    # Save uploaded image temporarily
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict button
    if st.button("Predict"):
        result, confidence = predict_pneumonia("temp_image.png")
        st.subheader(result)
        st.write(f"Prediction confidence: {confidence:.4f}")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & TensorFlow")
