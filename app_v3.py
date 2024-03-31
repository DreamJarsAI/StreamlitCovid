# Use pretrained models to create specialized web applications


# Import modules
import streamlit as st
import time
from PIL import Image, ImageEnhance
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification


# Create a function to read, process, and predict a CT lung scans image
def predict_img(image):
    # Call the processor and the model
    # Source: https://huggingface.co/DunnBC22/vit-base-patch16-224-in21k_covid_19_ct_scans
    # Image dataset: https://www.kaggle.com/datasets/luisblanche/covidct
    processor = AutoImageProcessor.from_pretrained("DunnBC22/vit-base-patch16-224-in21k_covid_19_ct_scans")
    model = AutoModelForImageClassification.from_pretrained("DunnBC22/vit-base-patch16-224-in21k_covid_19_ct_scans")

    # Process the image and convert to a tensor
    inputs = processor(images=image, return_tensors="pt")

    # Set the model to evaluation mode
    model.eval()

    # Use the model for predictions
    outputs = model(**inputs)

    # Obtain the predicted logits
    logits = outputs.logits.detach().numpy()

    # Get the predicted class
    classes = ['Covid-19 Positive', 'No Covid-19']
    predicted_class = classes[np.argmax(logits)]

    # Get the predicted probability based on logits
    predicted_probability = np.max(np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True), axis=1)[0]

    return predicted_class, predicted_probability


# Set page config
st.set_page_config(
    page_title="Covid-19 Detection Tool",  
    layout="centered", 
    initial_sidebar_state="auto")


# Define the main function
def main():
    # Set the title
    title_html = """
<div style="background-color:blue; padding:10px; text-align: center;">
<h1 style="color:white; margin:0;">Covid-19 Detection Tool</h1>
</div>
"""
    st.markdown(title_html, unsafe_allow_html=True)
    st.write("Upload a lung CT scan to check if it contains Covid-19 or not.")

    # Create a sidebar with image
    st.sidebar.image("coronavirus.jpeg", use_column_width=True)

    # Add a file uploader
    image_file = st.sidebar.file_uploader(
        "Upload an X-ray image (jpg, png, or jpeg)", 
        type=["jpg", "jpeg", "png"])
    
    # Check if an image was uploaded
    if image_file is not None:
        # Display the uploaded image
        our_image = Image.open(image_file)

        # Display the uploaded image
        if st.sidebar.button("Preview"):
            st.sidebar.image(our_image, use_column_width=True)
    
    # Add a dropdown menu to select different activities
    activities = ["Image Enhancement", "Diagnosis", "Disclaimer and Info"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    # Run the selected activity
    if choice == "Image Enhancement" and image_file is not None:
        st.subheader("Image Enhancement")

        # Create a radio button for image enhancement options
        enhance_type = st.sidebar.radio(
            "Enhance Type",
            ["Original", "Contrast", "Brightness"]
        )

        # Define the actions based on the selected option
        if enhance_type == "Contrast":
            # Create a slider to adjust the contrast
            c_rate = st.slider("Contrast Rate", 0.5, 5.0)

            # Enhance the contrast
            enhancer = ImageEnhance.Contrast(our_image)
            image_output = enhancer.enhance(c_rate)

            # Display the enhanced image
            st.image(image_output, width=600, use_column_width=True)

        elif enhance_type == "Brightness":
            # Create a slider to adjust the brightness
            b_rate = st.slider("Brightness Rate", 0.5, 5.0)

            # Enhance the brightness
            enhancer = ImageEnhance.Brightness(our_image)
            image_output = enhancer.enhance(b_rate)

            # Display the enhanced image
            st.image(image_output, width=600, use_column_width=True)
        
        else:
            st.text("Original Image")
            st.image(our_image, width=600, use_column_width=True)
  
    elif choice == "Diagnosis" and image_file is not None:
        st.subheader("Covid-19 Diagnosis")

        # Call the predict_img function
        predicted_class, predicted_probability = predict_img(our_image)

        # Create a progress bar to show the prediction progress
        progress_bar = st.progress(0)
        for percentage_complete in range(100):
            time.sleep(0.05)
            progress_bar.progress(percentage_complete + 1)
        
        # Display the prediction result
        st.text(f'Diagnosis: {predicted_class}, Predicted probability: {round(predicted_probability*100, 2)}%')
        
        # Add some warnings
        st.warning("Warning: This web app is just a demo. Please do not use it for medical diagnosis.")

    else:
        st.subheader("Disclaimer and Info")

        # Create a html template
        disclaimer_html = """
<div>
<h4>Disclaimer</h4>
<p><strong>This tool is for educational purposes only. It shoud not be used for clinical diagnosis.</strong></p>
<h4>Info</h4>
<p>The tool got inspiration from the following sources:</p>
<ul>
<li><a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7448820/">Coronavirus disease (COVID-19) detection in Chest X-Ray images using majority voting based classifier ensemble</a></li>
<li><a href="https://www.nature.com/articles/s41598-021-99015-3">Detection and analysis of COVID-19 in medical images using deep learning techniques</a></li>
<li><a href="https://www.bmj.com/content/370/bmj.m2426">The role of chest radiography in confirming covid-19 pneumonia</a></li>
</ul>
<p>More descriptions goes here...</p>
<p>The model suffers from the following limitations:</p>
<ul>
<li>Small dataset</li>
<li>Imags coming only from the Posterior-Anterior (PA) position</li>
<li>A fine-tuning is strongly recommended</li>
</ul>
</div>
"""

        # Display the disclaimer
        st.markdown(disclaimer_html, unsafe_allow_html=True)        


    # Add an About button
    if st.sidebar.button("About the author"):
        st.sidebar.subheader("Covid-19 Detection Tool")
        st.sidebar.markdown("Developed by: [Author's Name](https://github.com/yourusername)")
        st.sidebar.markdown("[author@gmail.com](mailto:author@gmail.com)")
        st.sidebar.text("All Rights Reserved (2024)")


# Run the main function
if __name__ == "__main__":
    main()