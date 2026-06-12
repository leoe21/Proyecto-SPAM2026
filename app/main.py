import streamlit as st
from model import load_model, predict_text

st.set_page_config(page_title="SMS Spam Demo", layout="centered")

st.title("SMS Spam Detection — Demo")

st.markdown("Enter an SMS message below and press Predict.")

model, model_type = load_model()

text = st.text_area("SMS text", height=150)

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter a message to classify.")
    else:
        label, score = predict_text(model, model_type, text)
        st.write("**Prediction:**", label)
        st.write("**Confidence:** {:.2%}".format(score))

st.markdown("---")
st.markdown("### Examples from dataset")
try:
    import random
    from pathlib import Path
    base = Path(__file__).resolve().parent.parent
    dataset_path = None
    for candidate in [base / 'SMSSpamCollection', base / 'data' / 'SMSSpamCollection']:
        if candidate.exists():
            dataset_path = candidate
            break
    if dataset_path:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        samples = random.sample(lines, min(5, len(lines)))
        for line in samples:
            parts = line.strip().split('\t', 1)
            if len(parts) == 2:
                label, msg = parts
                color = "🔴" if label == "spam" else "🟢"
                st.write(f"{color} **{label.upper()}**: {msg}")
    else:
        st.write("Dataset not found.")
except Exception:
    st.write("Could not load dataset examples.")
