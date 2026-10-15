import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# Map each model selection to its model file and image
MODEL_CONFIG = {
    "My_Model_0.1": {
        "model_path": "My_Model_0.1.keras",
        "image_path": "My_Model_0.1.png",
        "class_names": [
            "daisy",
            "dandelion",
            "rose",
            "sunflower",
            "tulip"
        ]
    },
    "My_Model_0.2": {
        "model_path": "My_Model_0.2.keras",
        "image_path": "My_Model_0.2.png",
        "class_names": [
            "daisy",
            "dandelion",
            "rose",
            "sunflower",
            "tulip"
        ]
    },
    "My_Model_1.0": {
        "model_path": "My_Model_1.0.keras",
        "image_path": "My_Model_1.0.png",
        "class_names": [
            "daisy",
            "dandelion",
            "rose",
            "sunflower",
            "tulip",
            "lily"
        ]
    }
}

# Sidebar
with st.sidebar:
    st.header("Settings")

    model_options = ["Select"] + list(MODEL_CONFIG.keys())

    selected_model = st.selectbox(
        "Select Model",
        model_options,
        index=0
    )

    if selected_model != "Select":
        config = MODEL_CONFIG[selected_model]

        st.image(
            config["image_path"],
            use_container_width=True
        )

        st.caption(f"Selected: {selected_model}")
    else:
        st.info("Please select a model.")

# Set configuration only after a model is selected
if selected_model != "Select":
    MODEL_PATH = MODEL_CONFIG[selected_model]["model_path"]
    CLASS_NAMES = MODEL_CONFIG[selected_model]["class_names"]


@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def predict_image(image, model, class_names):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = tf.keras.utils.img_to_array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array, verbose=0)

    if prediction.shape[-1] > 1:
        probabilities = prediction[0]
        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])

    else:
        probability = float(prediction[0][0])

        if probability >= 0.5:
            predicted_index = 1
            confidence = probability
        else:
            predicted_index = 0
            confidence = 1.0 - probability

    predicted_class = class_names[predicted_index]

    return predicted_class, confidence


st.title("🌸 Flower Classifier")

st.write(
    "Upload an image of a flower and the AI model "
    "will predict its class."
)

uploaded_file = st.file_uploader(
    "Choose a flower image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # Display a smaller image
    preview = image.copy()
    preview.thumbnail((250, 180))

    st.image(
        preview,
        caption="Uploaded image",
        width=200
    )

    if st.button("🔍 Predict Flower", type="primary"):
        try:
            model = load_model(MODEL_PATH)

            predicted_class, confidence = predict_image(
                image,
                model,
                CLASS_NAMES
            )

            st.success(
                f"Predicted Flower: **{predicted_class.title()}**"
            )

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

        except Exception as e:
            st.error("Unable to make the prediction.")
            st.exception(e)

st.divider()

st.caption("Powered by TensorFlow + Streamlit")

