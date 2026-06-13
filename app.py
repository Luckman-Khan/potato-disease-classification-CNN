from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "Models" / "potato_model" / "1"
CLASS_NAMES = ("Early Blight", "Late Blight", "Healthy")
DEMO_IMAGES = {
    "Early Blight": ROOT_DIR / "demo_images" / "early_blight.jpg",
    "Late Blight": ROOT_DIR / "demo_images" / "late_blight.jpg",
    "Healthy": ROOT_DIR / "demo_images" / "healthy.jpg",
}


st.set_page_config(
    page_title="Potato Leaf Health Predictor",
    page_icon=":material/eco:",
    layout="centered",
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    saved_model = tf.saved_model.load(str(MODEL_PATH))
    return saved_model.signatures["serving_default"]


def predict(image: Image.Image) -> tuple[str, float, np.ndarray]:
    model = load_model()
    rgb_image = image.convert("RGB").resize((256, 256))
    image_batch = np.expand_dims(np.asarray(rgb_image, dtype=np.float32), axis=0)
    output = model(input_layer=tf.convert_to_tensor(image_batch))
    probabilities = next(iter(output.values())).numpy()[0]
    predicted_index = int(np.argmax(probabilities))
    return (
        CLASS_NAMES[predicted_index],
        float(probabilities[predicted_index]),
        probabilities,
    )


def show_prediction(image: Image.Image, button_key: str) -> None:
    if not st.button(
        "Analyze leaf", type="primary", use_container_width=True, key=button_key
    ):
        return

    with st.spinner("Analyzing the leaf..."):
        predicted_class, confidence, probabilities = predict(image)

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Prediction</div>
            <div class="result-name">{predicted_class}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(confidence, text=f"Confidence: {confidence:.1%}")

    with st.expander("View all class probabilities"):
        for class_name, probability in zip(CLASS_NAMES, probabilities):
            st.write(f"**{class_name}:** {float(probability):.1%}")

    if predicted_class == "Healthy":
        st.success("The leaf appears healthy.")
    else:
        st.warning(
            "The model detected signs of disease. Consider confirming the "
            "result with a local agricultural specialist."
        )


def open_image(image_source) -> Image.Image:
    return Image.open(image_source).convert("RGB")


st.markdown(
    """
    <style>
        .block-container {max-width: 820px; padding-top: 2.5rem;}
        .hero {text-align: center; margin-bottom: 1.5rem;}
        .hero h1 {font-size: 2.4rem; margin-bottom: .35rem;}
        .hero p {color: #627067; font-size: 1.05rem;}
        .result-card {
            border: 1px solid #dce8df;
            border-radius: 16px;
            padding: 1.25rem 1.4rem;
            background: #f5fbf6;
            margin-top: .5rem;
        }
        .result-label {color: #52645a; font-size: .9rem; margin-bottom: .2rem;}
        .result-name {color: #176b36; font-size: 1.65rem; font-weight: 700;}
    </style>
    <div class="hero">
        <h1>Potato Leaf Health Predictor</h1>
        <p>Upload a photo, take one with your camera, or try a demo leaf to
        check for early blight, late blight, or a healthy leaf.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

upload_tab, camera_tab, demo_tab = st.tabs(
    ("Upload image", "Use camera", "Try demo images")
)

with upload_tab:
    uploaded_file = st.file_uploader(
        "Upload a potato leaf image",
        type=("jpg", "jpeg", "png", "webp"),
        help="For the best result, use a well-lit image with one leaf in focus.",
    )
    if uploaded_file is None:
        st.info("Choose or drag an image above to begin.")
    else:
        try:
            uploaded_image = open_image(uploaded_file)
            st.image(uploaded_image, caption=uploaded_file.name, use_container_width=True)
            show_prediction(uploaded_image, "analyze-upload")
        except UnidentifiedImageError:
            st.error("This file could not be read as an image. Please try another.")
        except Exception as error:
            st.error(f"Prediction failed: {error}")

with camera_tab:
    camera_photo = st.camera_input(
        "Take a photo of a potato leaf",
        help="Place one leaf in good light and keep the camera steady.",
    )
    if camera_photo is None:
        st.info("Allow camera access, then take a clear photo of the leaf.")
    else:
        try:
            captured_image = open_image(camera_photo)
            st.image(captured_image, caption="Camera photo", use_container_width=True)
            show_prediction(captured_image, "analyze-camera")
        except Exception as error:
            st.error(f"Prediction failed: {error}")

with demo_tab:
    demo_class = st.selectbox("Choose a demo leaf", CLASS_NAMES)
    demo_path = DEMO_IMAGES[demo_class]
    if not demo_path.exists():
        st.error(f"No demo image was found for {demo_class}.")
    else:
        try:
            demo_image = open_image(demo_path)
            st.image(
                demo_image,
                caption=f"Demo image: {demo_class}",
                use_container_width=True,
            )
            show_prediction(demo_image, "analyze-demo")
        except Exception as error:
            st.error(f"Could not load the demo image: {error}")

st.caption("This AI prediction is an aid and should not replace expert diagnosis.")
