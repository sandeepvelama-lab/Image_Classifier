import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import cv2
import numpy as np

st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🔍",
    layout="wide"
)

st.title("YOLO Computer Vision - Object Detection")
st.write("Upload an image to identify objects.")

@st.cache_resource
def load_model():
    try:
        model = YOLO("yolo26n.pt")
        return model
    except Exception as e:
        st.error(f"YOLO model loading failed: {e}")
        raise

model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "Confidence Threshold",
    0.1, 1.0, 0.25
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    results = model.predict(
        image,
        conf=confidence
    )

    annotated_image = results[0].plot()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("Detection Results")
        st.image(
            annotated_image[:, :, ::-1],
            use_container_width=True
        )

    detections = []

    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        conf = float(box.conf[0])

        detections.append({
            "Object": class_name,
            "Confidence": round(conf, 3)
        })

    st.subheader("Detected Objects")

    if detections:
        st.dataframe(
            pd.DataFrame(detections),
            use_container_width=True
        )
    else:
        st.info("No objects detected.")
