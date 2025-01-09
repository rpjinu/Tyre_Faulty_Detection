import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Set the title of the app
st.title("Tyre Fault Detection System")
st.write("Upload an image of a tire to check if it is faulty or good.")

# Load the saved model
@st.cache_resource  # Cache the model to avoid reloading on every interaction
def load_tyre_model():
    return load_model(r'C:\Users\Ranjan kumar pradhan\.vscode\tyre_faulty_detection\tyre_fault_detection_model.h5')

model = load_tyre_model()

# File uploader for the tire image
uploaded_file = st.file_uploader("Choose a tyre image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Tyre Image", use_column_width=True)

    # Preprocess the image
    img = image.load_img(uploaded_file, target_size=(224, 224))  # Resize to match model input size
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(img_array)

    # Interpret the prediction
    if prediction[0]< 0.5:
        st.write("**Prediction:** Faulty Tyre")
    else:
        st.write("**Prediction:** Good Tyre")

    # Optional: Display a confidence score
    st.write(f"**Confidence:** {prediction[0][0]:.2f}")

    # Optional: Visualize the image with the prediction
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis('off')
    if prediction[0] < 0.5:
        ax.set_title("Prediction: Faulty Tyre")
    else:
        ax.set_title("Prediction: Good Tyre")
    st.pyplot(fig)