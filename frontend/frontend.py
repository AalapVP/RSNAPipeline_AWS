import streamlit as st
import requests
from PIL import Image, ImageDraw
import io

API_URL = ""
st.set_page_config(page_title = "AI Radiologist", layout = "centered")
st.title("🫁 AI Radiologist Assistant")
st.markdown("Upload a chest X-Ray to check for lung opacity using an ensemble ViT/ResNet pipeline.")

#sensitivity slider
sensitivity = st.slider("Lower this to catch more cases (Higher Recall)", min_val = 0.0, max_val = 1.0, value = 0.35, step = 0.05)
uploaded_file = st.file_uploader("Upload X-Ray", type = ["jpg", "png", "jepg"])

if uploaded_file: 
    #display the uploaded image
    st.image(uploaded_file, caption = "Uploaded X-Ray", width = 300)
    
    if st.button("Run Diagnostics"):
        with st.spinner("Analysing via AWS Backend ... "):
            files = requests.post(f"{API_URL}?sensitivity={sensitivity}", files = files)
            if response.status_code == 200:
                data = response.json()
                st.subheader(f"Risk Score : {data['opacity_risk']:.2%}")
                
                if data['detection_triggered']:
                    st.warning("Pathalogy Detected! Analysing regions ...")
                    
                    # --- DRAWING BOXES ---
                    img_result = Image.open(uploaded_file).convert("RGB")
                    draw = ImageDraw(img_result)
                    
                    for box in data['boxes']:
                        draw.rectangle(box, outline = "red", width = 4)
                        
                    st.image(img_result, caption = "Detected Opacity Regions", use_container_width = True)
            else: 
                st.success("Normal Scan. No further detection required.")
    else:
        st.error(f"Backend Error: {response.status_code} - {response.text}")