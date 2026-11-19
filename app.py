
import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os

# ----------------------------------
# 1. Page configuration
# ----------------------------------

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide"
)

MODEL_PATH = "models/best_plant_disease_cnn.keras"
CLASS_PATH = "models/class_names.json"


# ----------------------------------
# 2. Load the trained CNN model
# ----------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_class_names():
    with open(CLASS_PATH, "r") as f:
        return json.load(f)


# ----------------------------------
# 3. Prediction function
# ----------------------------------

def predict_disease(image_bytes, model, class_names):

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32) / 255.0

    image = tf.expand_dims(image, axis=0)

    predictions = model.predict(image, verbose=0)[0]

    predicted_id = int(np.argmax(predictions))
    confidence = float(predictions[predicted_id])

    return class_names[predicted_id], confidence


# ----------------------------------
# 4. Application interface
# ----------------------------------

st.title("🌿 Plant Disease Detection System")

st.write(
    "Identify plant diseases using an image of a leaf "
    "and our trained Convolutional Neural Network."
)

st.divider()

try:
    model = load_model()
    class_names = load_class_names()
    st.success("CNN model loaded successfully!")
except Exception as e:
    st.error(f"Unable to load the model: {e}")
    st.stop()

left, right = st.columns(2)

with left:
    st.subheader("📤 Upload Plant Leaf")

    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        st.image(
            uploaded_file,
            caption="Uploaded plant leaf",
            use_container_width=True
        )

with right:
    st.subheader("🔍 Prediction Results")

    if uploaded_file:
        if st.button("Predict Disease", type="primary",
                     use_container_width=True):

            try:
                disease, confidence = predict_disease(
                    uploaded_file.getvalue(),
                    model,
                    class_names
                )

                st.success("Prediction completed!")

                st.write("### Predicted Disease")
                st.write(disease.replace("___", " — ").replace("_", " "))

                st.metric(
                    "Model Confidence",
                    f"{confidence * 100:.2f}%"
                )

                st.progress(confidence)

                st.caption(
                    "This is a model prediction, not a confirmed "
                    "agricultural diagnosis."
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    else:
        st.info("Upload a leaf image to begin.")

st.divider()

st.caption(
    "Plant Disease Detection | CNN-based Image Classification"
)
