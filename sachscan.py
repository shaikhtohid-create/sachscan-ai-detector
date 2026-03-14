import streamlit as st
from transformers import pipeline
from PIL import Image
import torch

# Page config
st.set_page_config(page_title="SachScan - AI Hai Ya Real?", layout="centered")

st.title("SachScan")
st.subheader("AI Generated hai ya Real photo? Upload karke check karo 🔥")

# Custom footer + hide default Streamlit stuff
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .custom-footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0e1117;
        color: #ffffff;
        text-align: center;
        padding: 12px 0;
        font-size: 14px;
        border-top: 1px solid #333;
        z-index: 999;
    }
    .custom-footer a {
        color: #4da6ff;
        text-decoration: none;
    }
    .custom-footer a:hover {
        text-decoration: underline;
    }
    </style>

    <div class="custom-footer">
        Made by <strong>Shaikh Tohid</strong> | 
        Contact: <a href="tel:9082647911">9082647911</a> | 
        Email: <a href="mailto:tohidempire7911@gmail.com">tohidempire7911@gmail.com</a>
    </div>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Model load (cache se fast rahega baad mein)
@st.cache_resource
def load_model():
    MODEL = "prithivMLmods/AI-vs-Deepfake-vs-Real"
    device = 0 if torch.cuda.is_available() else -1
    try:
        pipe = pipeline("image-classification", model=MODEL, device=device)
        return pipe
    except Exception as e:
        st.error(f"Model load fail: {e}")
        st.info("CPU mode pe chal raha hai, thoda time lagega.")
        pipe = pipeline("image-classification", model=MODEL, device=-1)
        return pipe

pipe = load_model()

# File uploader
uploaded_file = st.file_uploader(
    "Image yahan drag & drop karo ya click karke choose karo",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    # Image preview
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width=None)  # width=None = full width
    
    with st.spinner("AI model se check kar raha hoon... (pehli baar thoda time lagega)"):
        # Predict
        results = pipe(image)
        
        # Top result
        top = results[0]
        label = top['label']
        score = top['score']
        
        # AI probability calculate
        if label in ["Artificial", "Deepfake"]:
            ai_prob = round(score * 100, 1)
            real_prob = round((1 - score) * 100, 1)
        else:
            ai_prob = round((1 - score) * 100, 1)
            real_prob = round(score * 100, 1)
        
        # Verdict logic with Inconclusive
        if ai_prob >= 70:
            verdict = "Likely **AI-generated**"
            color = "red"
            conf = "High"
        elif ai_prob <= 30:
            verdict = "Likely **Real**"
            color = "green"
            conf = "High"
        else:
            verdict = "Inconclusive (possible real but some AI-like features)"
            color = "orange"
            conf = "Medium"
        
        # Result dikhana
        st.markdown(f"### Verdict: :{color}[{verdict}]")
        st.markdown(f"**AI Probability:** {ai_prob}%")
        st.markdown(f"**Real Probability:** {real_prob}%")
        st.markdown(f"**Confidence:** {conf}")
        
        st.progress(ai_prob / 100)
        
        # Extra details expander
        with st.expander("Model ne kya-kya dekha? (Detailed scores)"):
            for res in results:
                st.write(f"{res['label']}: {round(res['score']*100, 1)}%")
    
    st.caption("Disclaimer: Yeh sirf probability-based estimate hai. 100% guarantee nahi. Real-world mein thoda vary kar sakta hai.")

st.markdown("---")
st.caption("Powered by Hugging Face • Simple AI Image Detector")