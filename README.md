# SMS Spam Classification / Clasificación de SMS Spam

## ENGLISH — Project Overview

### 1) Project Description

**What it does:**  
This project develops and evaluates an SMS spam detection system using machine learning. It serves as a practical case study in cybersecurity and privacy-preserving machine learning techniques within the context of the ICESI "Hackeando la IA" course.

The system includes:
- **Exploratory Data Analysis (EDA)**: Statistical analysis of the SMSSpamCollection dataset
- **Baseline Model**: TF-IDF + Logistic Regression trained on SMS data (~97% accuracy)
- **Web Application**: Interactive Streamlit interface for real-time predictions
- **Security Considerations**: Documentation of model limitations, privacy implications, and path to privacy-preserving approaches

**Educational Focus:**  
This project demonstrates key concepts in ML security:
- **Data Privacy**: How centralized training exposes user data
- **Model Robustness**: Trade-offs between accuracy and fairness
- **Domain Mismatch**: Risks when using pre-trained models outside their training domain
- **Federated Learning & Differential Privacy**: Why decentralized, privacy-preserving approaches matter (see Security section)

### 2) Team Members

- Fabian A. Salazar Figueroa
- Luis E. Ordoñez Erazo
- Raúl A. Echeverry López

### 3) Dataset

- **Local**: `SMSSpamCollection` (5,572 messages: 87% ham, 13% spam)
- **Source**: [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- **Format**: Tab-separated (label \t message)
- **Privacy Note**: Contains real SMS messages; sensitive to privacy concerns

### 4) Models

#### Primary: sklearn Baseline (Recommended)
- **Type**: Pipeline (TF-IDF Vectorizer → Logistic Regression)
- **Training Data**: SMSSpamCollection (SMS-specific)
- **Performance**: 97% accuracy, 100% precision, 78.5% recall (high false negatives)
- **Advantages**: Fast, interpretable, trained on SMS domain
- **File**: `models/baseline_model.pkl`

#### Fallback: Hugging Face `Goodmotion/spam-mail-classifier`
- **Type**: Transformer-based text classifier
- **Training Data**: Email subjects (domain mismatch)
- **Performance**: Unknown on SMS; expect degradation
- **Advantages**: Handles new vocabulary via transfer learning
- **Disadvantages**: Domain mismatch (trained on email, not SMS)
- **Use Case**: Only if sklearn model unavailable

### 5) Project Structure

```
├── README.md                          # This file
├── requirements.txt                   # Python dependencies with versions
├── deliverable.txt                    # Repository link
├── SMSSpamCollection                  # Dataset (5,572 SMS messages)
│
├── app/
│   ├── main.py                        # Streamlit web interface
│   ├── model.py                       # Model loading & prediction logic
│   └── predict.py                     # Text preprocessing utilities
│
├── notebook/
│   └── analysis.ipynb                 # EDA, training, evaluation, security analysis
│
├── models/
│   └── baseline_model.pkl             # Trained sklearn baseline
│
└── scripts/
    ├── check_streamlit.py             # Health check for web app
    └── copy_dataset.py                # Dataset utility
```

### 6) Requirements

- **Python**: 3.10.16 (recommended for compatibility)
- **Dependencies**: See `requirements.txt`
- **Key Packages**:
  - `scikit-learn==1.9.0` (model training & inference)
  - `transformers==5.10.2` (HF fallback)
  - `streamlit==1.58.0` (web interface)
  - `pandas`, `numpy`, `matplotlib`, `seaborn` (EDA)

### 7) Setup & Installation

```bash
# Clone repository
git clone https://github.com/RaulEcheverryLopez/Proyecto-SPAM2026.git
cd Proyecto-SPAM2026

# Create virtual environment with Python 3.10.16
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 8) Running the Application

**Start the Streamlit web app:**
```bash
streamlit run app/main.py
```

**Access the app:**
- Local: http://localhost:8501
- Network: http://{your-ip}:8501

**Expected output:**
1. Model loads (sklearn preferred, HF fallback)
2. Sidebar shows which model is active
3. Enter SMS text and click "Predict"
4. View prediction (spam/ham) + confidence score
5. See random dataset examples at bottom

### 9) Exploring the Analysis

**Open the Jupyter notebook:**
```bash
jupyter notebook notebook/analysis.ipynb
```

**Sections covered:**
1. **Data Loading**: Locate and parse SMSSpamCollection
2. **Class Distribution**: Visualize spam/ham imbalance (87%/13%)
3. **Message Length Analysis**: Compare lengths by class
4. **Word Frequency**: Top words in spam vs. ham
5. **Baseline Model**: Train TF-IDF + Logistic Regression
6. **Evaluation**: Metrics (accuracy, precision, recall, F1, confusion matrix)
7. **Security Analysis**: Privacy implications and improvement paths

---

## 🔒 Security & Privacy Considerations

### Current State: Centralized Training

The current implementation is **fully centralized**:
- All data stored in `SMSSpamCollection` file
- Model trained on central server
- All predictions run locally or via public API

**Privacy Risks:**
- 📁 Central dataset is vulnerable to breaches
- 📊 Full data exposure if server is compromised
- 🔍 Model can be inverted to recover training data
- ⚖️ Compliance challenges (GDPR, privacy regulations)

### Improvement Path: Federated Learning + Differential Privacy

**Why this matters:**  
Modern systems (WhatsApp, Apple, Google) use **Federated Learning (FL)** + **Differential Privacy (DP)** to train models without centralizing user data.

**FL Approach:**
```
Local Device 1: Train locally → Send model update
Local Device 2: Train locally → Send model update
Local Device N: Train locally → Send model update
↓
Central Server: Aggregate updates → Improved model
```
**Benefit**: Data never leaves devices

**DP Addition:**
```
Model Update: [0.1, 0.05, -0.02, ...] 
+ Noise: [±0.001, ±0.001, ...]
= Noisy Update: [0.101, 0.049, -0.021, ...]
```
**Benefit**: Hides individual contributions, protects privacy

**This Project's Role:**  
- Demonstrates why FL/DP are necessary
- Baseline for comparing FL/DP performance
- Educational case study for privacy-preserving ML

### Known Limitations

| Aspect | Current | Limitation |
|--------|---------|-----------|
| **Training** | Centralized | All data in one place |
| **Privacy** | None | No differential privacy |
| **Distribution** | Local/cloud | Not federated |
| **Domain** | SMS | HF fallback trained on email |
| **Recall** | 78.5% | High false negatives (missed spam) |
| **Access** | Rate limited (100/hr per session) | No authentication; rate limit resets on server restart |
| **Monitoring** | Basic logging | No production-grade monitoring |

### Recommendations for Production

1. **Privacy**:
   - Implement Federated Learning for distributed training
   - Apply Differential Privacy for noise injection
   - Use secure aggregation for model updates

2. **Security**:
   - Add authentication (API keys, OAuth)
   - Implement rate limiting
   - Monitor for adversarial inputs
   - Log all predictions for audit trail

3. **Robustness**:
   - Detect adversarial examples
   - Implement anomaly detection
   - Monitor model drift
   - Update models regularly

4. **Fairness**:
   - Test for bias by language/region
   - Ensure equitable performance across demographics
   - Document limitations clearly

---

## 🎓 Learning Outcomes

By completing this project, students will:

✅ **[OT1] Develop automated cybersecurity solutions** using Python + ML libraries  
✅ **[OT3] Describe security elements** across ML project lifecycle  
✅ Understand **data privacy** and centralization risks  
✅ Recognize importance of **Federated Learning + Differential Privacy**  
✅ Build production-ready **web applications** for ML models  
✅ Apply **monitoring and logging** for transparency  

---

## 📚 References & Further Reading

- [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- [Hugging Face Model](https://huggingface.co/Goodmotion/spam-mail-classifier)
- [The Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
- [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873)
- [OWASP: Machine Learning Security Top 10](https://owasp.org/www-project-machine-learning-security-top-10/)

---

## ESPAÑOL — Descripción del Proyecto

### 1) Descripción del Proyecto

**Qué hace:**  
Este proyecto desarrolla y evalúa un sistema de detección de spam en SMS utilizando aprendizaje automático. Sirve como caso de estudio práctico en ciberseguridad y técnicas de aprendizaje automático que preservan la privacidad, dentro del contexto del curso "Hackeando la IA" de ICESI.

El sistema incluye:
- **Análisis Exploratorio de Datos (EDA)**: Análisis estadístico del dataset SMSSpamCollection
- **Modelo Base**: TF-IDF + Logistic Regression entrenado en SMS (~97% accuracy)
- **Aplicación Web**: Interfaz Streamlit interactiva para predicciones en tiempo real
- **Consideraciones de Seguridad**: Documentación de limitaciones, implicaciones de privacidad, y camino hacia enfoques que preserven privacidad

**Enfoque Educativo:**  
Este proyecto demuestra conceptos clave en seguridad de ML:
- **Privacidad de Datos**: Cómo el entrenamiento centralizado expone datos de usuarios
- **Robustez del Modelo**: Trade-offs entre accuracy y equidad
- **Domain Mismatch**: Riesgos de usar modelos preentrenados fuera de su dominio
- **Federated Learning y Differential Privacy**: Por qué los enfoques descentralizados importan

### 2) Integrantes

- Fabian A. Salazar Figueroa
- Luis E. Ordoñez Erazo
- Raúl A. Echeverry López

### 3) Dataset

- **Local**: `SMSSpamCollection` (5,572 mensajes: 87% ham, 13% spam)
- **Fuente**: [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)

### 4) Estructura del Proyecto

- **Raíz**: README.md, requirements.txt, SMSSpamCollection, deliverable.txt
- **`app/`**: Aplicación Streamlit (main.py, model.py, predict.py)
- **`notebook/`**: Jupyter notebook con EDA, entrenamiento y evaluación
- **`models/`**: Modelo baseline guardado (baseline_model.pkl)
- **`scripts/`**: Scripts auxiliares

### 5) Configuración e Instalación

```bash
git clone https://github.com/RaulEcheverryLopez/Proyecto-SPAM2026.git
cd Proyecto-SPAM2026

python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
source .venv/bin/activate      # macOS/Linux

pip install -r requirements.txt
```

### 6) Ejecutar la Aplicación

```bash
streamlit run app/main.py
```

Accede a: http://localhost:8501

### 7) Notebook

```bash
jupyter notebook notebook/analysis.ipynb
```

Incluye: EDA, entrenamiento del baseline, evaluación y análisis de seguridad.

---

## 🔒 Consideraciones de Seguridad y Privacidad

Ver sección en inglés arriba (Security & Privacy Considerations) para detalles completos.

**Resumen**: El proyecto actual es centralizado. Muestra por qué **Federated Learning + Differential Privacy** son necesarios para sistemas de producción.

---

## Notes

- All code is production-ready with error handling, logging, and documentation
- Dataset is publicly available (UCI repository)
- Model achieves good performance on SMS domain (baseline)
- Project aligns with ICESI course on AI security ("Hackeando la IA")


