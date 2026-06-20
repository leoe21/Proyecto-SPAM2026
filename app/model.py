from pathlib import Path
import joblib
import logging
from typing import Tuple, Any, Optional

logger = logging.getLogger(__name__)


def _validate_model(model: Any, model_type: str) -> bool:
    """Validate that the loaded model can make a prediction."""
    try:
        test_input = "test message"
        if model_type == 'sklearn':
            model.predict([test_input])
        else:
            model(test_input, truncation=True)
        return True
    except Exception as e:
        logger.warning(f"Model validation failed for {model_type}: {e}")
        return False


def load_model() -> Tuple[Optional[Any], Optional[str]]:
    """Load spam classification model with priority fallback.

    Priority:
      1. `models/baseline_model.pkl` (sklearn pipeline trained on SMSSpamCollection)
      2. Hugging Face `Goodmotion/spam-mail-classifier` (fallback, domain mismatch warning)

    Returns:
      (model, model_type): model object and type string ('sklearn', 'hf'), or (None, None) if both fail.

    Note:
      - sklearn model is preferred (trained on SMS data, less domain mismatch)
      - HF is fallback with domain mismatch warning (trained on emails, not SMS)
      - Explicit logging for debugging and reproducibility
    """
    repo_root = Path(__file__).resolve().parents[1]
    candidate = repo_root / 'models' / 'baseline_model.pkl'

    if candidate.exists():
        try:
            model = joblib.load(candidate)
            if _validate_model(model, 'sklearn'):
                logger.info(f"Loaded sklearn baseline from {candidate}")
                return model, 'sklearn'
            else:
                logger.warning(f"sklearn model validation failed, trying fallback")
        except FileNotFoundError:
            logger.error(f"Model file not found at {candidate}")
        except PermissionError:
            logger.error(f"Permission denied reading {candidate}")
        except Exception as e:
            logger.error(f"Failed to load sklearn model: {type(e).__name__}: {e}")

    logger.info("Falling back to Hugging Face model (domain mismatch warning: trained on emails, not SMS)")
    try:
        from transformers import pipeline
        model = pipeline(
            "text-classification",
            model="Goodmotion/spam-mail-classifier",
            tokenizer="Goodmotion/spam-mail-classifier"
        )
        if _validate_model(model, 'hf'):
            logger.info("Loaded Hugging Face model successfully")
            return model, 'hf'
        else:
            logger.warning("HF model validation failed")
    except ImportError:
        logger.error("transformers library not installed (required for HF fallback)")
    except Exception as e:
        logger.error(f"Failed to load HF model: {type(e).__name__}: {e}")

    logger.critical("Failed to load any model - both sklearn and HF failed")
    return None, None


def predict_text(model: Optional[Any], model_type: Optional[str], text: str) -> Tuple[str, float]:
    """Generate spam/ham prediction with confidence score.

    Both model types receive raw text:
      - sklearn Pipeline: TfidfVectorizer handles its own tokenization (trained on raw text)
      - HF pipeline: transformer tokenizer handles its own tokenization
    Applying simple_preprocess before the sklearn model would strip digits (phone numbers,
    prize amounts) that the TF-IDF vocabulary learned during training — causing missed spam.

    Args:
      model: Loaded model object (sklearn Pipeline or transformers pipeline)
      model_type: 'sklearn' or 'hf'
      text: Input SMS message

    Returns:
      (label, confidence): Tuple of prediction label and confidence [0.0, 1.0]
    """
    if model is None or model_type is None:
        logger.warning("predict_text called with None model")
        return 'UNKNOWN', 0.0

    if not text.strip():
        logger.debug("Input is empty")
        return 'UNKNOWN', 0.0

    if model_type == 'sklearn':
        return _predict_sklearn(model, text)
    elif model_type == 'hf':
        return _predict_hf(model, text)
    else:
        logger.error(f"Unknown model_type: {model_type}")
        return 'UNKNOWN', 0.0


def _predict_sklearn(model: Any, text: str) -> Tuple[str, float]:
    """Predict using sklearn pipeline."""
    try:
        label = model.predict([text])[0]

        confidence = 0.0
        if hasattr(model, 'predict_proba'):
            proba_all = model.predict_proba([text])[0]
            confidence = _extract_spam_probability(model, proba_all)

        logger.debug(f"sklearn prediction: {label} ({confidence:.2%})")
        return label, confidence

    except Exception as e:
        logger.error(f"sklearn prediction failed: {type(e).__name__}: {e}")
        return 'UNKNOWN', 0.0


def _normalize_hf_label(raw_label: str) -> str:
    """Map HF model raw labels to canonical 'spam'/'ham' strings."""
    normalized = raw_label.lower().strip()
    if normalized in ('spam', '1', 'label_1'):
        return 'spam'
    if normalized in ('ham', '0', 'label_0', 'not spam'):
        return 'ham'
    return raw_label


def _predict_hf(model: Any, text: str) -> Tuple[str, float]:
    """Predict using Hugging Face pipeline."""
    try:
        output = model(text, truncation=True)

        if isinstance(output, list) and len(output) > 0:
            result = output[0]
            raw_label = result.get('label', 'UNKNOWN')
            score = float(result.get('score', 0.0))
            label = _normalize_hf_label(raw_label)
            logger.debug(f"HF prediction: {raw_label} → {label} ({score:.2%})")
            return label, score

        logger.warning("Unexpected HF model output format")
        return 'UNKNOWN', 0.0

    except Exception as e:
        logger.error(f"HF prediction failed: {type(e).__name__}: {e}")
        return 'UNKNOWN', 0.0


def _extract_spam_probability(model: Any, proba_all: tuple) -> float:
    """Extract spam class probability from sklearn pipeline probabilities."""
    try:
        if hasattr(model, 'named_steps') and 'clf' in model.named_steps:
            clf = model.named_steps['clf']
        else:
            clf = model

        classes = getattr(clf, 'classes_', None)

        if classes is not None and 'spam' in classes:
            spam_idx = list(classes).index('spam')
            return float(proba_all[spam_idx])

        return float(max(proba_all))

    except (IndexError, ValueError, AttributeError) as e:
        logger.warning(f"Could not extract spam probability: {e}, using max")
        return float(max(proba_all)) if proba_all else 0.0

