import streamlit as st
from PIL import Image
import base64
import requests
from groq_api_key import groq_api_key
from deep_translator import GoogleTranslator

import io
import re

# ========== Medical Glossary ========== #
medical_glossary = {
    "cerebral atrophy": "Shrinkage or loss of brain cells, often related to aging or diseases like Alzheimer’s.",
    "lesion": "An area of abnormal tissue, which could be caused by disease or injury.",
    "infarct": "Tissue death due to lack of blood supply, often seen in strokes.",
    "edema": "Swelling caused by excess fluid trapped in tissues.",
    "hemorrhage": "Excessive bleeding, either inside or outside the body.",
    "calcification": "Build-up of calcium in body tissues, often hardening them.",
    "contrast enhancement": "Technique using contrast agents in imaging to highlight areas, often related to inflammation or tumors.",
    "ventricular dilation": "Enlargement of the brain's ventricles, may suggest hydrocephalus or brain atrophy.",
    "mass effect": "Pressure from a mass (like a tumor) displacing surrounding brain structures.",
    "midline shift": "A shift of brain structures from their normal position, usually due to swelling or mass."
}

# ========== Explain Medical Terms ========== #
def highlight_and_explain_terms(text):
    explanations = {}
    lower_text = text.lower()
    for term in medical_glossary:
        if term in lower_text:
            explanations[term] = medical_glossary[term]
    return explanations

# ========== Extract Text from Base64 Image Using OCR.Space API ========== #
def extract_text_from_base64_image(base64_img):
    url = "https://api.ocr.space/parse/image"
    payload = {
        'base64Image': 'data:image/png;base64,' + base64_img,
        'language': 'eng',
        'isOverlayRequired': False,
        'apikey': 'K88392357288957'
    }
    response = requests.post(url, data=payload)
    result = response.json()
    try:
        return result['ParsedResults'][0]['ParsedText'].strip()
    except Exception as e:
        return f"OCR failed: {e}"

# ========== Encode image to base64 ========== #
def encode_image(image_file):
    return base64.b64encode(image_file.read()).decode("utf-8")

# ========== Generate LLM Response from Groq ========== #
def generate_response_groq(text, language='English'):
    prompt = f"Please analyze the following medical image description and provide insights in {language}:\n\n{text}"
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer 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",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1000,
        "temperature": 0.7
    }
    response = requests.post(url, headers=headers, json=payload)
    response_json = response.json()
    if 'choices' in response_json:
        return response_json['choices'][0]['message']['content']
    else:
        st.error(f"Unexpected response format: {response_json}")
        return "❌ Analysis failed due to an unexpected response from the API."

# ========== Translate response if needed ========== #
def translate_response(text, target_language):
    if target_language.lower() == "hindi":
        try:
            translated = GoogleTranslator(source='auto', target='hi').translate(text)
            return translated
        except Exception as e:
            st.error(f"Translation failed: {e}")
            return text
    return text


# ========== Streamlit UI ========== #
st.set_page_config(page_title="Visual Medical Assistant", page_icon="🩺", layout="wide")
st.title("Visual Medical Assistant 👨‍⚕️ 🩺 🏥")
st.subheader("An app to help with medical image analysis")

# Language Selection
language = st.radio("Choose language for analysis:", ["English", "Hindi"])

# Upload Image
file_uploaded = st.file_uploader('Upload the medical image', type=['png', 'jpg', 'jpeg'])

if file_uploaded:
    st.image(file_uploaded, width=250, caption="Uploaded Image")
    submit = st.button("Generate Analysis")

    if submit:
        with st.spinner("Analyzing the image..."):
            try:
                base64_img = encode_image(file_uploaded)
                image_text = extract_text_from_base64_image(base64_img)

                MAX_CHARS = 3000
                if len(image_text) > MAX_CHARS:
                    image_text = image_text[:MAX_CHARS] + "\n\n[Text truncated for model input]"

                response_text = generate_response_groq(image_text, language=language)
                final_output = translate_response(response_text, language)

                st.markdown("### 🧾 Analysis Report")
                st.write(final_output)

                st.markdown("---")
                st.markdown("### 🧠 Medical Term Explanations")
                explanations = highlight_and_explain_terms(image_text)

                if explanations:
                    for term, definition in explanations.items():
                        st.markdown(f"**{term.capitalize()}**: {definition}")
                else:
                    st.info("No complex medical terms found for explanation.")

            except Exception as e:
                st.error(f"❌ Error occurred: {str(e)}")
