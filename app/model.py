from pathlib import Path
import joblib
from typing import Tuple, Any

from predict import simple_preprocess

_model = None
_model_type = None


def load_model() -> Tuple[Any, Any]:
    """Load and cache a model.

    Priority:
      1. `models/baseline_model.pkl` (joblib sklearn pipeline)
      2. Hugging Face `Goodmotion/spam-mail-classifier` pipeline

    Returns (model, meta) where `meta` is None for HF pipeline or a string
    identifying the model type for sklearn-based models.
    """
    global _model, _model_type
    if _model is not None:
        return _model, _model_type

    repo_root = Path(__file__).resolve().parents[1]
    candidate = repo_root / 'models' / 'baseline_model.pkl'
    if candidate.exists():
        try:
            _model = joblib.load(candidate)
            _model_type = 'sklearn'
            return _model, _model_type
        except Exception:
            _model = None
            _model_type = None

    # fallback: transformers pipeline
    if _model is None:
        try:
            from transformers import pipeline
            MODEL_NAME = "Goodmotion/spam-mail-classifier"
            _model = pipeline("text-classification", model=MODEL_NAME, tokenizer=MODEL_NAME)
            _model_type = 'hf'
        except Exception:
            _model = None
            _model_type = None

    return _model, _model_type


def predict_text(model, model_type, text: str) -> Tuple[str, float]:
    """Return (label, confidence) for a single text using the loaded model.

    For sklearn pipelines we return the predicted label and the probability
    for the `spam` class when available.
    For Hugging Face pipelines we keep existing behavior.
    """
    if model is None:
        return 'UNKNOWN', 0.0

    # Keep preprocessing explicit and aligned with notebook tokenization.
    cleaned_text = simple_preprocess(text)
    if not cleaned_text:
        return 'UNKNOWN', 0.0

    if model_type == 'sklearn':
        try:
            # sklearn Pipeline: get predicted label
            pred = model.predict([cleaned_text])[0]
            proba = None
            if hasattr(model, 'predict_proba'):
                proba_all = model.predict_proba([cleaned_text])[0]
                # try to get class order from final estimator
                try:
                    clf = model.named_steps.get('clf') if hasattr(model, 'named_steps') else model
                    classes = getattr(clf, 'classes_', None)
                except Exception:
                    classes = None
                if classes is not None:
                    # find spam index if present
                    try:
                        spam_idx = list(classes).index('spam')
                        proba = float(proba_all[spam_idx])
                    except ValueError:
                        # spam not in classes -> take max prob
                        proba = float(max(proba_all))
                else:
                    proba = float(max(proba_all))
            return pred, proba if proba is not None else 0.0
        except Exception:
            return 'UNKNOWN', 0.0

    # assume HF pipeline
    try:
        out = model(cleaned_text, truncation=True)
        if isinstance(out, list) and out:
            res = out[0]
            label = res.get('label')
            score = float(res.get('score', 0.0))
            return label if label is not None else 'UNKNOWN', score
    except Exception:
        pass
    return 'UNKNOWN', 0.0
