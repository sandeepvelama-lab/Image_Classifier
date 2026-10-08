import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

IMAGE_SIZE = (180, 180)

# IMPORTANT:
# Replace these with the EXACT output of:
# print(dataset.class_names)

CLASS_NAMES = [
    "daisy",
    "dandelion",
    "rose",
    "sunflower",
    "tulip",
]

MODEL_PATH = "flower_photos.keras"


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def predict_image(image, model):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = tf.keras.utils.img_to_array(image)
    image_array = image_array / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array, verbose=0)

    # Multi-class model
    if prediction.shape[-1] > 1:
        probabilities = prediction[0]

        predicted_index = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_index])

    # Binary model
    else:
        probability = float(prediction[0][0])

        if probability >= 0.5:
            predicted_index = 1
            confidence = probability
        else:
            predicted_index = 0
            confidence = 1.0 - probability

    predicted_class = CLASS_NAMES[predicted_index]

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

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded image",
        use_container_width=True
    )

    if st.button("🔍 Predict Flower", type="primary"):

        try:
            model = load_model()

            predicted_class, confidence = predict_image(
                image,
                model
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
