# CardioAI: Intelligent Health Analytics System
An advanced clinical decision support system integrating XGBoost, A* Search, and Rule-based Inference to provide explainable cardiac risk stratification.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.6%2B-green.svg)](https://xgboost.readthedocs.io/)
[![Flask](https://img.shields.io/badge/Flask-2.2%2B-orange.svg)](https://flask.palletsprojects.com/)
[![Vanilla JS](https://img.shields.io/badge/Vanilla%20JS-ES6%2B-yellow.svg)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)

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

## 📂 Professional Repository Structure
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
├── notebooks/
│   ├── phase_1.ipynb                 # EDA, Preprocessing, and ML Training
│   └── CardioAI_Phase2_Clean.ipynb   # AI Architecture (PEAS, Search, KB)
└── docs/
    ├── project_report.md             # Complete FYP technical report
    ├── system_guide.md               # Master system summary & study roadmap
    ├── viva_questions.md             # Folder guide & technical viva questions
    ├── project_alignment.md          # Assignment requirement mapping
    ├── justification.md              # clinical & technical decision justifications
    ├── kb_predicate_logic.md         # 24 rules represented in Predicate Logic (FOPL)
    ├── ml_pipeline.md                # Phase 1 ML processing details
    └── ai_algorithms.md              # Phase 2 planning and reasoning details
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

## 📚 Deep Documentation
For examiners or developers wishing to dive into the technical theory and rubric fulfillment, please review the files in the `docs/` folder:
* **[project_report.md](file:///c:/Users/m/Downloads/CardioAI-main/CardioAI-main/docs/project_report.md)**: Full-length Final Year Project style technical report.
* **[system_guide.md](file:///c:/Users/m/Downloads/CardioAI-main/CardioAI-main/docs/system_guide.md)**: Master system summary explaining all three pages, the R-Simulator, and a study guide.
* **[viva_questions.md](file:///c:/Users/m/Downloads/CardioAI-main/CardioAI-main/docs/viva_questions.md)**: Graded technical viva preparation questions (ML, AI, Full-Stack).
* **[kb_predicate_logic.md](file:///c:/Users/m/Downloads/CardioAI-main/CardioAI-main/docs/kb_predicate_logic.md)**: Formal mathematical representations of the 24 clinical rules in First-Order Predicate Logic.

---

## 🙏 Acknowledgments
CardioAI leverages open access datasets. Special thanks to the **UCI Machine Learning Repository** for providing the foundational Cleveland, Hungarian, Swiss, and Long Beach Heart Disease datasets.

*Made with ❤️ by MAFA. Not for Clinical Use.*
