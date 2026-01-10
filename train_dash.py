import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Apple Freshness Classifier",
    page_icon="🍎",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #2E7D32;
        padding: 20px 0;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
    }
    .fresh {
        background-color: #E8F5E9;
        border: 2px solid #4CAF50;
    }
    .stale {
        background-color: #FFF3E0;
        border: 2px solid #FF9800;
    }
    </style>
""", unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_model():
    model_path = "model/fresh_vs_spoiled_cnn.h5"
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        st.error(f"Model not found at {model_path}. Please train the model first.")
        return None

# Preprocess image
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Main app
def main():
    st.markdown('<h1 class="main-header">🍎 Apple Freshness Classifier</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load the model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # File uploader
    st.subheader("Upload an Apple Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of an apple"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Predict button
        if st.button("🔍 Classify Apple", use_container_width=True):
            with st.spinner("Analyzing..."):
                # Preprocess and predict
                processed_img = preprocess_image(image)
                prediction = model.predict(processed_img, verbose=0)[0][0]
                
                # Interpret results
                # Assuming 0 = Fresh, 1 = Stale (adjust based on your class_names)
                confidence = float(prediction * 100)
                
                if prediction < 0.5:
                    label = "Fresh 🍏"
                    result_class = "fresh"
                    confidence_display = f"{100 - confidence:.2f}%"
                else:
                    label = "Stale 🍂"
                    result_class = "stale"
                    confidence_display = f"{confidence:.2f}%"
                
                # Display results
                st.markdown("---")
                st.markdown(f'<div class="prediction-box {result_class}">', unsafe_allow_html=True)
                st.markdown(f"### Prediction: **{label}**")
                st.markdown(f"### Confidence: **{confidence_display}**")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Progress bar for visualization
                st.progress(confidence / 100)
    
    else:
        st.info("👆 Please upload an image to get started")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("This app classifies apples as **Fresh** or **Stale** using a MobileNetV2-based CNN.")
        
        st.markdown("---")
        st.header("📊 Model Info")
        st.write("- **Architecture**: MobileNetV2")
        st.write("- **Input Size**: 224x224")
        st.write("- **Classes**: Fresh / Stale")
        
        st.markdown("---")
        st.header("💡 Tips")
        st.write("- Use clear, well-lit images")
        st.write("- Ensure the apple is the main subject")
        st.write("- Avoid blurry or dark images")

if __name__ == "__main__":
    main()
