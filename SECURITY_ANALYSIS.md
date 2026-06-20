# Security & Privacy Analysis — SMS Spam Detection Project

**Context**: This document supplements the Jupyter notebook with security analysis relevant to ICESI's "Hackeando la IA" course.

---

## 1. Current Architecture: Centralized Model Training

### Data Flow

```
SMSSpamCollection (local file)
        ↓
    [EDA & Analysis] ← notebook/analysis.ipynb
        ↓
[Train TF-IDF + LogReg] ← sklearn.Pipeline
        ↓
models/baseline_model.pkl (221 KB)
        ↓
[Streamlit App] ← app/main.py (predictions in real-time)
        ↓
User predictions (logged)
```

### Security Risks in Centralized Training

| Risk | Severity | Description |
|------|----------|-------------|
| **Data Breach** | 🔴 HIGH | All 5,572 SMS stored in single unencrypted file |
| **No Access Control** | 🔴 HIGH | Anyone with repo access can read all SMS |
| **Pickle Deserialization (RCE)** | 🔴 HIGH | `joblib.load()` executes arbitrary code if `.pkl` is replaced by an attacker — model supply chain attack (OWASP ML Top 10: ML06) |
| **Model Inversion** | 🟠 MEDIUM | Attacker can query model to recover training data |
| **No Audit Trail** | 🟠 MEDIUM | No logging of who accessed data or when |
| **Privacy Violation** | 🟠 MEDIUM | SMS contains PII (phone numbers, personal info) |

---

## 2. Model Robustness Analysis

### 2.1 Recall Gap: 78.5% (33 False Negatives)

**Finding**: The sklearn baseline misses 22% of actual spam messages.

```
Confusion Matrix (test set, n=1115):
                Predicted
Actual     Ham    Spam
Ham        966     0      ← Perfect (no false positives)
Spam        33    116     ← Missing 33 spam (false negatives)
```

**Impact**:
- ❌ Users receive malicious SMS without warning
- ✅ Very low false positive rate (good for UX)
- ⚠️ Trade-off: Choose precision or recall based on use case

**Improvement Strategy**:
- Lower classification threshold (from 0.5 to 0.3)
- Use ensemble methods (Random Forest, XGBoost)
- Implement active learning (human-in-the-loop)

### 2.2 Domain Mismatch: HF Fallback Risk

**Finding**: The HF model (fallback) was trained on emails, not SMS.

```
Baseline (sklearn): Trained on SMS → ~97% accuracy on SMS (expected)
Fallback (HF):      Trained on emails → Unknown accuracy on SMS (expected lower)
```

**Risk**: If sklearn fails silently, app uses HF without user knowledge.

**Mitigation**:
- ✅ Refactored `model.py` to log which model loads
- ✅ App shows sidebar indicating active model
- ✅ Explicit warning if HF is active

### 2.3 No Adversarial Robustness

**Finding**: Model NOT tested against adversarial inputs.

**Adversarial Examples**:
```
Original: "FREE MONEY CALL NOW"
Evasion:  "FR33 M0N3Y C4LL N0W"
          "FREE ... MONEY ... CALL ... NOW" (space insertion)
          "FrEe mOnEy CaLl NoW" (mixed case)
```

**Real-world impact**: Spammers use these techniques. Model will fail.

**Improvement**:
- Implement adversarial training
- Robustness evaluation (FGSM, PGD attacks)
- Character normalization in preprocessing

---

## 3. Privacy-Preserving Solutions: Federated Learning + Differential Privacy

### 3.1 Problem: Centralized Training Violates Privacy

**Current state**: All SMS data in `SMSSpamCollection` → Single point of failure

**Compliance risk**:
- ❌ GDPR: Not processing data minimally (storing all SMS)
- ❌ CCPA: Users cannot opt-out
- ❌ Local regulations: SMS might be regulated communications

### 3.2 Solution 1: Federated Learning (FL)

**How it works**:

```
Step 1: Central server initializes model
        model_v0 → Device 1, Device 2, Device 3

Step 2: Each device trains on LOCAL data (never sent to server)
        Device 1: local_SMS → update_1
        Device 2: local_SMS → update_2
        Device 3: local_SMS → update_3

Step 3: Aggregator merges updates → improved model
        model_v1 = Aggregate(update_1, update_2, update_3)
        
Step 4: Repeat until convergence
```

**Benefits**:
- ✅ Data never leaves devices (privacy by design)
- ✅ GDPR compliant (user retains control)
- ✅ Scales to millions of devices
- ✅ Inherent robustness (no single point of failure)

**Trade-offs**:
- ⚠️ Slower convergence (more communication rounds)
- ⚠️ Higher communication costs
- ⚠️ More complex to implement

**Real-world examples**:
- 🍎 Apple: Federated learning for keyboard prediction
- 📱 Google: Federated learning for keyboard (Gboard)
- 📨 WhatsApp: Uses secure aggregation + FL for spam detection

### 3.3 Solution 2: Differential Privacy (DP)

**Problem with FL alone**: Server aggregates model updates → can invert to recover data

```
Attack: Invert aggregated updates → recover individual training data
```

**DP Solution**: Add noise to updates before aggregation

```
Individual update:  [0.1, 0.05, -0.02, 0.03, ...]
+ Gaussian noise:   [±0.001, ±0.001, ±0.001, ...]
= Noisy update:     [0.101, 0.049, -0.021, 0.031, ...]
```

**Mathematical guarantee**: (ε, δ)-differential privacy

```
Pr[Aggregate(D) ∈ S] ≤ e^ε × Pr[Aggregate(D') ∈ S] + δ

Intuition: Query output similar whether you're in dataset or not
```

**Benefits**:
- ✅ Provable privacy guarantee (mathematical proof)
- ✅ Scales to arbitrary number of users
- ✅ Robust against inference attacks

**Trade-offs**:
- ⚠️ Added noise reduces model accuracy
- ⚠️ Requires careful tuning of ε (privacy budget)
- ⚠️ More complex to implement

### 3.4 Combined: FL + DP Architecture

```
┌─────────────────────────────────────────┐
│         Central Server                   │
│  (No access to raw data)                │
│  ├─ Model parameters v0                 │
│  ├─ Aggregation logic                   │
│  └─ Privacy accounting (ε, δ tracking)  │
└─────────────────────────────────────────┘
         ↓        ↓        ↓
    ┌────────┬────────┬────────┐
    ↓        ↓        ↓
  Device 1  Device 2  Device N
  ────────  ────────  ────────
  Local SMS Local SMS Local SMS
    ↓        ↓        ↓
  [Train]  [Train]  [Train]
    ↓        ↓        ↓
  Update 1 Update 2 Update N
    + noise  + noise  + noise
    ↓        ↓        ↓
  └────────┬────────┬────────┘
           ↓
    [Aggregate with DP]
           ↓
     model_v1 (improved, private)
```

**Benefits of combined approach**:
- ✅ Data privacy (FL)
- ✅ User privacy (DP noise)
- ✅ GDPR/CCPA compliant
- ✅ Scales to billions of users

---

## 4. Implementation: Current vs. Future

### Current Implementation (Centralized)

```python
# Centralized: Data + training in one place
df = pd.read_csv('SMSSpamCollection')  # Load all 5,572 SMS
model = Pipeline([...]).fit(df['text'], df['label'])  # Train
joblib.dump(model, 'baseline_model.pkl')  # Save
```

**Security posture**: Vulnerable to data breaches

### Future Implementation (FL + DP)

```python
# Federated + Differential Privacy
from tensorflow_federated import learning

# Each device does this:
def device_update(local_data, model_params):
    # Train on local data only
    local_model = model.fit(local_data)
    update = local_model.get_weights() - model_params
    
    # Add DP noise
    noise = np.random.normal(0, sigma, update.shape)
    noisy_update = update + noise
    
    return noisy_update

# Server aggregates:
def server_aggregate(updates, learning_rate):
    avg_update = np.mean(updates, axis=0)  # Average of noisy updates
    model_params += learning_rate * avg_update
    return model_params
```

**Security posture**: Privacy-preserving by design

---

## 5. Security Checklist for Production

### Data Protection
- [ ] Encrypt data at rest (AES-256)
- [ ] Encrypt data in transit (TLS 1.3)
- [ ] Implement federated learning (no centralized data)
- [ ] Apply differential privacy (ε, δ tracking)
- [ ] Data retention policy (auto-delete after X days)

### Model Security
- [ ] Version control for models
- [ ] Integrity checks (SHA-256 hashes)
- [ ] Monitor for model drift
- [ ] Adversarial robustness testing
- [ ] Regular model retraining

### Access Control
- [ ] Authentication (API keys, OAuth 2.0)
- [ ] Authorization (role-based access control)
- [ ] Rate limiting (100 requests/hour baseline)
- [ ] IP whitelisting
- [ ] Audit logging (all accesses)

### Monitoring & Incidents
- [ ] Real-time monitoring of predictions
- [ ] Alert on anomalous patterns
- [ ] Incident response plan
- [ ] Regular security audits
- [ ] Penetration testing

---

## 6. Learning Outcomes

By studying this analysis, students will understand:

✅ **Why centralization is risky**: Single point of failure for privacy  
✅ **How FL works**: Distributed training without centralizing data  
✅ **How DP works**: Mathematical privacy guarantees via noise  
✅ **Trade-offs**: Privacy vs. accuracy, scalability vs. complexity  
✅ **Real-world applications**: WhatsApp, Apple, Google use these techniques  

---

## 7. References

### Papers
- [Federated Learning: Challenges, Methods, and Future Directions](https://arxiv.org/abs/1908.07873)
- [The Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
- [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)

### Tools & Frameworks
- [TensorFlow Federated](https://tensorflow.org/federated)
- [OpenMined: Privacy-preserving ML](https://openmined.org/)
- [Opacus: PyTorch Differential Privacy](https://opacus.ai/)

### Standards & Compliance
- [GDPR: General Data Protection Regulation](https://gdpr.eu/)
- [CCPA: California Consumer Privacy Act](https://oag.ca.gov/privacy/ccpa)
- [OWASP ML Security Top 10](https://owasp.org/www-project-machine-learning-security-top-10/)

---

## 8. Conclusion

This project demonstrates **why privacy-preserving ML matters**:

1. **Current state** (centralized) = vulnerable to data breaches
2. **Future state** (FL + DP) = provably private and secure
3. **Real-world impact** = billions of users rely on these techniques

The SMS spam detection case shows both the power of ML (97% accuracy) and the necessity of privacy-preserving approaches for ethical, compliant systems.

**Key takeaway**: Security and privacy are not optional—they're fundamental to responsible AI.

---

---

# ESPAÑOL — Análisis de Seguridad y Privacidad

**Contexto**: Este documento complementa el Jupyter notebook con el análisis de seguridad relevante para el curso "Hackeando la IA" de ICESI.

---

## 1. Arquitectura Actual: Entrenamiento Centralizado

### Flujo de Datos

```
SMSSpamCollection (archivo local)
        ↓
    [EDA & Análisis] ← notebook/analysis.ipynb
        ↓
[Entrenar TF-IDF + LogReg] ← sklearn.Pipeline
        ↓
models/baseline_model.pkl (221 KB)
        ↓
[App Streamlit] ← app/main.py (predicciones en tiempo real)
        ↓
Predicciones del usuario (registradas en logs)
```

### Riesgos de Seguridad en el Entrenamiento Centralizado

| Riesgo | Severidad | Descripción |
|--------|-----------|-------------|
| **Brecha de Datos** | 🔴 ALTO | Los 5.572 SMS almacenados en un único archivo sin cifrar |
| **Sin Control de Acceso** | 🔴 ALTO | Cualquiera con acceso al repositorio puede leer todos los SMS |
| **Deserialización Pickle (RCE)** | 🔴 ALTO | `joblib.load()` ejecuta código arbitrario si el `.pkl` es reemplazado por un atacante — ataque a la cadena de suministro del modelo (OWASP ML Top 10: ML06) |
| **Inversión del Modelo** | 🟠 MEDIO | Un atacante puede consultar el modelo para recuperar datos de entrenamiento |
| **Sin Registro de Auditoría** | 🟠 MEDIO | No hay log de quién accedió a los datos ni cuándo |
| **Violación de Privacidad** | 🟠 MEDIO | Los SMS contienen PII (números de teléfono, información personal) |

---

## 2. Análisis de Robustez del Modelo

### 2.1 Brecha de Recall: 78.5% (33 Falsos Negativos)

**Hallazgo**: El baseline sklearn falla en el 22% de los mensajes spam reales.

```
Matriz de Confusión (conjunto de prueba, n=1115):
                Predicho
Real       Ham    Spam
Ham        966     0      ← Perfecto (sin falsos positivos)
Spam        33    116     ← 33 spam no detectados (falsos negativos)
```

**Impacto**:
- ❌ Usuarios reciben SMS maliciosos sin advertencia
- ✅ Tasa de falsos positivos muy baja (buena experiencia de usuario)
- ⚠️ Trade-off: elegir precisión o recall según el caso de uso

**Estrategia de Mejora**:
- Bajar el umbral de clasificación (de 0.5 a 0.3)
- Usar métodos de ensemble (Random Forest, XGBoost)
- Implementar aprendizaje activo (humano en el ciclo)

### 2.2 Desajuste de Dominio: Riesgo del Fallback HF

**Hallazgo**: El modelo HF (fallback) fue entrenado con emails, no con SMS.

```
Baseline (sklearn): Entrenado en SMS → ~97% accuracy en SMS (esperado)
Fallback (HF):      Entrenado en emails → Accuracy desconocida en SMS (esperada menor)
```

**Riesgo**: Si sklearn falla silenciosamente, la app usa HF sin que el usuario lo sepa.

**Mitigación**:
- ✅ `model.py` registra en logs qué modelo se carga
- ✅ La app muestra en la barra lateral cuál modelo está activo
- ✅ Advertencia explícita si HF está activo

### 2.3 Sin Robustez Adversarial

**Hallazgo**: El modelo NO fue probado contra entradas adversariales.

**Ejemplos Adversariales**:
```
Original: "FREE MONEY CALL NOW"
Evasión:  "FR33 M0N3Y C4LL N0W"
          "FREE ... MONEY ... CALL ... NOW" (inserción de espacios)
          "FrEe mOnEy CaLl NoW" (mayúsculas mezcladas)
```

**Impacto real**: Los spammers usan estas técnicas. El modelo fallará.

**Mejora**:
- Implementar entrenamiento adversarial
- Evaluación de robustez (ataques FGSM, PGD)
- Normalización de caracteres en el preprocesamiento

---

## 3. Soluciones que Preservan la Privacidad: Federated Learning + Differential Privacy

### 3.1 Problema: El Entrenamiento Centralizado Viola la Privacidad

**Estado actual**: Todos los SMS en `SMSSpamCollection` → Punto único de fallo

**Riesgo de cumplimiento**:
- ❌ GDPR: No procesa datos de forma mínima (almacena todos los SMS)
- ❌ CCPA: Los usuarios no pueden optar por no participar
- ❌ Regulaciones locales: Los SMS pueden ser comunicaciones reguladas

### 3.2 Solución 1: Federated Learning (FL)

**Cómo funciona**:

```
Paso 1: El servidor central inicializa el modelo
        modelo_v0 → Dispositivo 1, Dispositivo 2, Dispositivo 3

Paso 2: Cada dispositivo entrena con datos LOCALES (nunca enviados al servidor)
        Dispositivo 1: SMS_local → actualización_1
        Dispositivo 2: SMS_local → actualización_2
        Dispositivo 3: SMS_local → actualización_3

Paso 3: El agregador combina las actualizaciones → modelo mejorado
        modelo_v1 = Agregar(actualización_1, actualización_2, actualización_3)

Paso 4: Repetir hasta convergencia
```

**Beneficios**:
- ✅ Los datos nunca salen de los dispositivos (privacidad por diseño)
- ✅ Cumple con GDPR (el usuario mantiene el control)
- ✅ Escala a millones de dispositivos
- ✅ Robustez inherente (sin punto único de fallo)

**Ejemplos del mundo real**:
- 🍎 Apple: Federated Learning para predicción de teclado
- 📱 Google: Federated Learning para teclado (Gboard)
- 📨 WhatsApp: Usa agregación segura + FL para detección de spam

### 3.3 Solución 2: Differential Privacy (DP)

**Problema con FL solo**: El servidor agrega actualizaciones del modelo → puede invertirlas para recuperar datos individuales

**Solución DP**: Agregar ruido a las actualizaciones antes de la agregación

```
Actualización individual: [0.1, 0.05, -0.02, 0.03, ...]
+ Ruido gaussiano:        [±0.001, ±0.001, ±0.001, ...]
= Actualización con ruido: [0.101, 0.049, -0.021, 0.031, ...]
```

**Garantía matemática**: privacidad diferencial (ε, δ)

**Beneficios**:
- ✅ Garantía de privacidad demostrable (prueba matemática)
- ✅ Escala a número arbitrario de usuarios
- ✅ Robusto contra ataques de inferencia

---

## 4. Lista de Verificación de Seguridad para Producción

### Protección de Datos
- [ ] Cifrar datos en reposo (AES-256)
- [ ] Cifrar datos en tránsito (TLS 1.3)
- [ ] Implementar Federated Learning (sin datos centralizados)
- [ ] Aplicar Differential Privacy (seguimiento de ε, δ)

### Seguridad del Modelo
- [ ] Control de versiones para modelos
- [ ] Verificaciones de integridad (hashes SHA-256)
- [ ] Monitorear drift del modelo
- [ ] Pruebas de robustez adversarial
- [ ] Reentrenamiento regular del modelo

### Control de Acceso
- [ ] Autenticación (claves API, OAuth 2.0)
- [ ] Autorización (control de acceso basado en roles)
- [ ] Limitación de tasa (100 solicitudes/hora de base)
- [ ] Lista blanca de IPs
- [ ] Registro de auditoría (todos los accesos)

### Monitoreo e Incidentes
- [ ] Monitoreo en tiempo real de predicciones
- [ ] Alertas en patrones anómalos
- [ ] Plan de respuesta a incidentes
- [ ] Auditorías de seguridad regulares
- [ ] Pruebas de penetración

---

## 5. Conclusión

Este proyecto demuestra **por qué el ML que preserva la privacidad es importante**:

1. **Estado actual** (centralizado) = vulnerable a brechas de datos
2. **Estado futuro** (FL + DP) = privado y seguro de forma demostrable
3. **Impacto real** = miles de millones de usuarios dependen de estas técnicas

El caso de detección de spam en SMS muestra tanto el poder del ML (97% de accuracy) como la necesidad de enfoques que preserven la privacidad para sistemas éticos y conformes con la normativa.

**Conclusión clave**: La seguridad y la privacidad no son opcionales — son fundamentales para una IA responsable.
