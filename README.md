# CardioAI: Intelligent Health Analytics System
An advanced clinical decision support system integrating XGBoost, A* Search, and Rule-based Inference to provide explainable cardiac risk stratification.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.6%2B-green.svg)](https://xgboost.readthedocs.io/)
[![Flask](https://img.shields.io/badge/Flask-2.2%2B-orange.svg)](https://flask.palletsprojects.com/)
[![Vanilla JS](https://img.shields.io/badge/Vanilla%20JS-ES6%2B-yellow.svg)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)


<div align="center">
    <img src="frontend/WhatsApp%20Image%202026-06-27%20at%2010.27.34%20AM.jpeg" alt="Dashboard Preview" width="90%">
    <br>
    <h3><i>"Hybrid AI system for explainable cardiac risk prediction."</i></h3>
</div>
<br>

---

## 📖 Overview
CardioAI was built to bridge the gap between raw machine learning probabilities and human-understandable clinical pathways. Instead of just giving a percentage risk, CardioAI explains why a patient is at risk, what to do about it, and how specific lifestyle changes impact their outlook.

It processes 13 clinical features (from the combined 920-record UCI Heart Disease dataset) through five distinct AI subsystems:
1. **XGBoost Classifier**: Predicts heart disease probability with high accuracy (~0.898 AUC).
2. **A* Search Algorithm**: Finds the optimal, minimum-cost clinical pathway to safe patient management.
3. **Forward Chaining Knowledge Base**: Uses 24 medical rules (AHA/ACC sourced) to derive symbolic clinical conclusions.
4. **Counterfactual Simulator**: Probes the ML model in real-time to generate sensitivity analyses.
5. **TF-IDF Search Engine**: Queries an integrated clinical knowledge base for relevant medical literature.

---

## 📂 Repository Structure
The project has been professionally organized to separate concerns across the backend, frontend, models, and research notebooks.

```
CardioAI/
├── backend/
│   ├── models/
│   │   ├── heart_disease_model.pkl   # Serialized production XGBoost Model
│   │   └── heart_scaler.pkl          # Fitted StandardScaler
│   └── server.py                     # Flask REST API Server
├── frontend/
│   ├── index.html                    # Main Single Page App structure
│   ├── styles.css                    # Dark Theme CSS with CSS grid layout
│   ├── app.js                        # Main UI Controller & API client logic
│   ├── algorithms.js                 # Client-side A*, KB, and NLP algorithms
│   └── model.js                      # Offline JS ML Fallback logic
└── notebooks/
    ├── phase_1.ipynb                 # EDA, Preprocessing, and ML Training
    └── CardioAI_Phase2_Clean.ipynb   # AI Architecture (PEAS, Search, KB)
```

---

## 🚀 Quick Start Guide
To run CardioAI on your local machine, you need Python 3.10+ installed.

### 1. Clone & Setup
```bash
git clone https://github.com/MAFA-KHAN/CardioAI.git
cd CardioAI
pip install -r requirements.txt
```

### 2. Start the Backend API
```bash
cd backend
python server.py
```
*(The server will start on http://127.0.0.1:5000)*

### 3. Launch the Frontend
We run a local server to handle CORS permissions when fetching API endpoints:
```bash
cd ../frontend
python -m http.server 8000
```
*Open your browser and navigate to `http://localhost:8000` to interact with the dashboard.*

---

## 🧠 System Architecture

### 1. Machine Learning Pipeline
Trained on the combined 920-record UCI dataset, the system uses 4 engineered interaction features (`bp_chol_interaction`, `exercise_stress_score`, `thalch_age_ratio`, `age_risk_group`) to capture complex medical relationships. The XGBoost model incorporates L1/L2 regularization and operates at a calibrated **`0.35` sensitivity threshold** to prioritize catching false negatives, yielding a clinical recall of **`93.1%`**.

### 2. Intelligent Search Agent (PEAS)
The system represents the clinical care journey as a state-space graph. Based on the patient's vitals, an **A* Search Algorithm** uses an admissible heuristic to chart the exact clinical interventions required (e.g., `ECG Evaluation` $\rightarrow$ `Cardiology Consultation` $\rightarrow$ `Risk Stratification`).

### 3. Knowledge Base Inference
A forward-chaining rule engine applies **24 specific medical rules** derived from AHA, ACC, and NHLBI guidelines. It transforms numeric vitals into symbolic facts (e.g., BP $\ge 140$ $\rightarrow$ Hypertension) and derives multi-condition conclusions to ensure safety nets are in place.

### 4. Dual-Mode Fallback (Edge AI)
If the Flask backend is unreachable, the frontend automatically intercepts requests and routes them to local JS scripts. It uses a mathematical sigmoid approximation of the XGBoost coefficients in `model.js` and local heap-queue implementations in `algorithms.js` to execute the A* pathfinding and forward-chaining rules entirely within the browser.

### 5. Real-time Simulator
The Simulator tab allows users to select preset patient profiles inspired by K-Means cluster centroids. It runs live sensitivity analyses, showing precisely which lifestyle changes (e.g., reducing ST depression) will most effectively lower risk.

---

## 🛠️ How CardioAI Was Built
CardioAI didn't start as a finished hybrid system — it was built in layers, each one added to fix a specific gap in the last:

1. **Phase 1 — Data & ML Core.** Started from the raw 920-record UCI Heart Disease dataset (4 merged sub-datasets with different missingness patterns). Cleaned and unified the schema, then engineered four interaction features (`bp_chol_interaction`, `exercise_stress_score`, `thalch_age_ratio`, `age_risk_group`) to capture compound medical risk that raw vitals alone don't expose. Trained and tuned an XGBoost classifier on top, deliberately lowering the decision threshold from 0.50 to 0.35 — a direct response to optimizing for recall rather than accuracy, since a missed at-risk patient is clinically worse than a false alarm.
2. **Phase 2 — Symbolic Reasoning Layer.** Once the ML model could output a risk score, the next gap was *interpretability* — a number alone isn't a clinical decision. This is where the project shifted from a prediction task into an **agent architecture**: an A* search agent to plan the optimal sequence of clinical interventions, and a forward-chaining knowledge base (24 rules from AHA/ACC/NHLBI guidelines) to independently verify the ML output symbolically, so the system never silently trusts the model alone.
3. **Counterfactual & Retrieval Layer.** Built a simulator on top of the trained model to answer "what-if" questions live (e.g. what happens to risk if ST depression drops), plus a TF-IDF retrieval layer so every conclusion could be tied back to a literature reference instead of floating as an unexplained number.
4. **Full-Stack Wrap.** Wrapped the whole pipeline in a Flask REST API with a vanilla-JS frontend — and, critically, a JS-side fallback (`model.js`, `algorithms.js`) that reimplements the XGBoost decision boundary and the search/KB logic natively in-browser, so the system degrades gracefully instead of breaking if the backend goes down.

The throughline across all four phases: every addition was justified by a specific weakness in the previous layer — accuracy → recall, prediction → explainability, static output → interactive probing, backend-dependent → backend-optional — rather than features bolted on for their own sake.

---



## 🙏 Acknowledgments
CardioAI leverages open access datasets. Special thanks to the **UCI Machine Learning Repository** for providing the foundational Cleveland, Hungarian, Swiss, and Long Beach Heart Disease datasets.

*Made with ❤️ by MAFA. Not for Clinical Use.*
