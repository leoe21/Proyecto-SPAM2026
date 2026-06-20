import streamlit as st
import logging
import uuid
import random
import zipfile
import io
import urllib.request
from datetime import datetime
from pathlib import Path
from model import load_model, predict_text

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter to prevent abuse of the spam detection model."""

    def __init__(self, max_requests: int = 100, window_seconds: int = 3600):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}

    def is_allowed(self, session_id: str) -> bool:
        """Check if request is allowed for this session."""
        now = datetime.now()

        recent = [
            ts for ts in self.requests.get(session_id, [])
            if (now - ts).total_seconds() < self.window_seconds
        ]

        if len(recent) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for session {session_id}")
            return False

        recent.append(now)
        self.requests[session_id] = recent

        # Purge sessions with no recent activity to prevent unbounded memory growth
        stale = [sid for sid, ts_list in self.requests.items() if not ts_list]
        for sid in stale:
            del self.requests[sid]

        return True


rate_limiter = RateLimiter(max_requests=100, window_seconds=3600)

_UCI_DATASET_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
)


def _find_or_download_dataset(base: Path) -> Path | None:
    """Return path to SMSSpamCollection, downloading from UCI if needed."""
    for candidate in [base / 'SMSSpamCollection', base / 'data' / 'SMSSpamCollection']:
        if candidate.exists():
            return candidate

    destination = base / 'SMSSpamCollection'
    try:
        with urllib.request.urlopen(_UCI_DATASET_URL, timeout=30) as response:
            data = response.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for name in zf.namelist():
                if 'SMSSpamCollection' in name and not name.endswith('/'):
                    destination.write_bytes(zf.read(name))
                    logger.info(f"Dataset downloaded from UCI to {destination}")
                    return destination
    except Exception as e:
        logger.error(f"Dataset download failed: {e}")

    return None

st.set_page_config(page_title="SMS Spam Detection", layout="centered")

st.title("📱 SMS Spam Detection")
st.markdown("Classify SMS messages as spam or legitimate (ham) using machine learning.")

model, model_type = load_model()

if model is None:
    st.error(
        "❌ **Critical Error**: No spam detection model available.\n\n"
        "Please ensure:\n"
        "1. `models/baseline_model.pkl` exists (sklearn baseline) OR\n"
        "2. Internet connection for Hugging Face model download\n"
        "3. `transformers` library is installed"
    )
    logger.critical("No model loaded - app non-functional")
    st.stop()

with st.sidebar:
    st.header("ℹ️ Model Information")
    if model_type == 'sklearn':
        st.success(f"✅ **Active Model**: sklearn Baseline")
        st.info(
            "**Trained on**: SMS messages (SMSSpamCollection)\n\n"
            "**Algorithm**: TF-IDF + Logistic Regression\n\n"
            "**Performance**: ~97% accuracy, 100% precision (low recall)"
        )
    elif model_type == 'hf':
        st.warning(f"⚠️ **Active Model**: Hugging Face Fallback")
        st.warning(
            "**Domain Mismatch**: This model was trained on **email**, not SMS.\n\n"
            "Expect lower performance on SMS. For best results, use the sklearn baseline.\n\n"
            "[View model](https://huggingface.co/Goodmotion/spam-mail-classifier)"
        )
    else:
        st.error("Unknown model type")

    st.divider()
    st.subheader("🔒 Security Notice")
    st.caption(
        "This model is for **demonstration only**. "
        "Production systems should implement rate limiting, authentication, and monitoring."
    )

st.markdown("---")
st.subheader("🔍 Prediction")

text = st.text_area("Enter SMS message to classify:", height=120, placeholder="Type SMS here...")

col1, col2 = st.columns(2)

with col1:
    predict_btn = st.button("🔎 Predict", use_container_width=True)

with col2:
    clear_btn = st.button("🗑️ Clear", use_container_width=True)

if clear_btn:
    st.rerun()

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if predict_btn:
    session_id = st.session_state.session_id

    if not rate_limiter.is_allowed(session_id):
        st.error(
            "⚠️ **Rate limit exceeded**\n\n"
            "You have exceeded the maximum number of predictions (100/hour). "
            "Please try again later."
        )
        logger.warning(f"Rate limit hit for session {session_id}")
        st.stop()

    if not text.strip():
        st.warning("⚠️ Please enter a message to classify.")
        logger.warning("User submitted empty text")
    else:
        with st.spinner("Classifying message..."):
            label, confidence = predict_text(model, model_type, text)

        st.divider()

        if label == 'UNKNOWN':
            st.error("❌ **Classification failed**. Please try again or contact support.")
            logger.error(f"Prediction failed for text: {text[:50]}...")
        else:
            col1, col2 = st.columns(2)

            with col1:
                if label.lower() == 'spam':
                    st.error(f"🚫 **{label.upper()}**")
                else:
                    st.success(f"✅ **{label.upper()}**")

            with col2:
                st.metric("Confidence", f"{confidence:.1%}")

            st.caption(f"*Model: {model_type.upper()} | Confidence: {confidence:.2%}*")

            logger.info(
                f"Prediction: label={label}, confidence={confidence:.2%}, "
                f"model={model_type}, text_len={len(text)}, session={session_id}"
            )

st.markdown("---")
st.subheader("📊 Dataset Examples")

try:
    base = Path(__file__).resolve().parent.parent

    with st.spinner("Loading dataset examples..."):
        dataset_path = _find_or_download_dataset(base)

    if dataset_path:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        if lines:
            samples = random.sample(lines, min(5, len(lines)))
            st.caption(f"*Random examples from SMSSpamCollection ({len(lines)} total messages)*")

            for line in samples:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    label, msg = parts
                    emoji = "🚫" if label.lower() == "spam" else "✅"
                    st.text(f"{emoji} [{label.upper():4}] {msg[:160]}")
        else:
            st.warning("Dataset is empty")
    else:
        st.warning("📁 Dataset not found and could not be downloaded. Check your internet connection.")
        logger.warning("Dataset unavailable: not found locally and download failed")

except Exception as e:
    st.warning(f"Could not load dataset examples: {e}")
    logger.exception("Failed to load dataset examples")

st.markdown("---")
st.markdown(
    """
    **About this project**: Educational implementation for SMS spam classification.
    Part of ICESI's "Hackeando la IA" course on cybersecurity in machine learning.

    **Dataset**: [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
    """
)

