import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
import torch.nn.functional as F

# STEP 1: Configure Streamlit app settings
st.set_page_config(page_title="AI Image Classifier", layout="centered")
st.title("🖼️ CPU-Based Computer Vision Web App")
st.write("This app uses a pre-trained ResNet18 model to classify images.")

# STEP 2: Import the required libraries
# shown above

# STEP 3: Configure the application to run only on CPU settings
device = torch.device('cpu')

# STEP 4 & 5: Load ResNet18 and recommended preprocessing
@st.cache_resource
def initialize_model():
    # Load weights metadata to get labels and transforms automatically
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.to(device)
    model.eval()  # Set to evaluation mode

    # Extract human-readable labels and preprocessing steps
    categories = weights.meta["categories"]
    preprocess_steps = weights.transforms()
    return model, categories, preprocess_steps

model, labels, preprocess = initialize_model()

# STEP 6: User interface for image upload
uploaded_file = st.file_uploader("📤 Upload an image (JPG or PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the input image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # STEP 7: Convert image to tensor and perform inference
    input_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)

        # STEP 8: Apply Softmax and find Top-5 classes
        probabilities = F.softmax(output, dim=1)[0]
        top5_prob, top5_catid = torch.topk(probabilities, 5)

        # Get the top prediction details for the dynamic result line
        top_label = labels[top5_catid[0]]
        top_confidence = top5_prob[0].item() * 100

        # Prepare DataFrames for visualization and table
        chart_data = pd.DataFrame({
            "Category": [labels[catid] for catid in top5_catid],
            "Confidence": [prob.item() for prob in top5_prob]
        })

        # Display the most probable result line
        st.subheader("Final Result ✨")
        st.success(f"The uploaded image is most probably a **{top_label}** (Confidence: {top_confidence:.2f}%).")

        # STEP 9: Visualization
        st.subheader("📊 Top-5 Predictions Visualization")
        # Passing numerical data to the bar chart
        st.bar_chart(chart_data.set_index("Category"))

        # Display table with formatted strings for the user
        st.table(chart_data.assign(Confidence=chart_data['Confidence'].map(lambda x: f"{x * 100:.2f}%")))

        # STEP 10: Discussion of results, level, and process path
        with st.expander("🔍 Click to view Technical Discussion"):
            st.subheader("1. Level of Recognition 📈")
            st.write("""
                This application operates at the **Image Classification** level, assigning a 
                single category label to the entire uploaded image. It utilizes **Transfer Learning** by leveraging a pre-trained **ResNet18** model. This landmark architecture uses 
                **residual connections** to allow for deep feature learning without 
                signal degradation and effectively identify 1,000 different object categories 
                based on universal visual features.
                """)

            st.subheader("2. The Process Path (How the AI Works) 👣")
            st.write("""
                The application follows a modern **Computer Vision Pipeline** that mimics 
                human sight:
                * **Input & Sensing:** The uploader captures the image, which the computer 
                  represents as a **matrix of numerical pixel data**.
                * **Pre-processing:** The system applies **Intensity Normalization** and 
                  resizing. This scales pixel values to a standard range, helping the neural 
                  network converge faster and remain robust to variations.
                * **Feature Extraction:** Using **Convolutional Layers**, the model 
                  automatically extracts hierarchical features. Early layers detect simple 
                  **edges**, while deeper layers identify complex **shapes and textures**.
                * **Interpreting & Output:** The CPU (acting as the **Interpreting Device**) 
                  processes these features. A **Softmax function** is applied to the final 
                  layer to map internal relationships into human-readable probabilities.
                """)

else:
    st.info("👆 Please upload an image to start classification.")

st.markdown("---")
st.caption("BSD3513 Introduction to Artificial Intelligence | Lab 5 – Computer Vision")
