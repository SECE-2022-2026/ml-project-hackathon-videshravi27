import streamlit as st
import rasterio
import numpy as np
import tensorflow as tf
from tempfile import NamedTemporaryFile

# Load your trained model
MODEL_PATH = 'satellite_image_classification_model.keras'
model = tf.keras.models.load_model(MODEL_PATH)

# Define the categories for predictions
categories = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

def load_geotiff_image(image_path, target_size=(64, 64)):
    """
    Load a GeoTIFF image, extract the first three bands (assumed to be RGB),
    resize it, and preprocess it for prediction.
    """
    with rasterio.open(image_path) as src:
        if src.count < 3:
            raise ValueError("The provided GeoTIFF file must have at least 3 bands for RGB.")
        
        # Read the first three bands (RGB)
        img_data = src.read([1, 2, 3])  # Extract RGB bands
        img_data = np.moveaxis(img_data, 0, -1)  # Rearrange to (height, width, channels)

        # Normalize the pixel values to [0, 1] range
        img_data = img_data / 255.0

        # Resize the image to the model's input size
        img_resized = tf.image.resize(img_data, target_size)
        return img_resized.numpy()

# Streamlit app setup
st.title("Satellite Image Classification")
st.write("Upload a `.tif` file to classify its land use category.")

# File uploader
uploaded_file = st.file_uploader("Upload a GeoTIFF file (.tif)", type=['tif'])

if uploaded_file is not None:
    try:
        # Save the uploaded file to a temporary file
        with NamedTemporaryFile(delete=False, suffix='.tif') as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        
        # Load and preprocess the image
        st.write("Processing the uploaded image...")
        image = load_geotiff_image(temp_file_path)

        # Add a batch dimension for prediction
        image_batch = np.expand_dims(image, axis=0)

        # Predict the class
        predictions = model.predict(image_batch)
        predicted_class = categories[np.argmax(predictions)]

        # Display the results
        st.success(f"Predicted Class: {predicted_class}")
        st.bar_chart(predictions[0])  # Optional: Show the prediction probabilities

    except Exception as e:
        st.error(f"An error occurred: {e}")