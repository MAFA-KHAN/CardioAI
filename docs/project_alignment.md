# CardioAI — Academic Requirements Alignment

This document outlines how the CardioAI implementation aligns with the academic requirements for the 4th Semester AI & ML Combined Project.

---

## 1. Requirement & Implementation Mapping

### A. Supervised Machine Learning (Phase 1)
* **Academic Criteria**: Data exploration, cleaning, model comparison, class balancing, serialization, and deployment.
* **Code Implementation**:
  * **Exploration & Preprocessing**: `notebooks/phase_1.ipynb` (handling missing value zero-replacements).
  * **Model Training & Comparison**: `notebooks/phase_1.ipynb` (comparing Logistic Regression, Random Forests, and XGBoost).
  * **Class Balancing**: Balanced training set using SMOTE.
  * **Serialization**: Models serialized as `backend/models/heart_disease_model.pkl` and `backend/models/heart_scaler.pkl`.
  * **API Deployment**: Served via `backend/server.py` using Flask to expose the `/predict` and `/simulate` endpoints.

### B. Intelligent Search Agent (Phase 2)
* **Academic Criteria**: Formal state-space problem formulation, admissible heuristic definition ($h(n)$), and A* Search path optimization.
* **Code Implementation**:
  * **Problem Formulation & PEAS**: Defined in `notebooks/CardioAI_Phase2_Clean.ipynb` (Section 2) and `docs/ai_algorithms.md`.
  * **Dynamic Graph Construction**: `backend/server.py` (lines 245–264) and `frontend/algorithms.js` (lines 53–75) build patient-specific nodes based on clinical inputs.
  * **Admissible Heuristic**: Stored in Python dictionary (`server.py` lines 79–93) and JS object (`algorithms.js` lines 27–42).
  * **A* Search Execution**: Finds the lowest-cost treatment path using a priority queue. Implemented in `backend/server.py` (lines 222–242) and `frontend/algorithms.js` (lines 80–111).

### C. Rule-Based Expert System (Phase 2)
* **Academic Criteria**: Knowledge base definition containing at least 20 IF-THEN rules, forward-chaining inference engine, and a reasoning trace output.
* **Code Implementation**:
  * **Knowledge Base Rules**: Contains 24 medical rules mapped to guidelines (AHA, ACC, NHLBI). Defined in `backend/server.py` (lines 280–315) and `frontend/algorithms.js` (lines 142–177).
  * **Forward Chaining Engine**: Computes new clinical facts by checking subset conditions. Implemented in `backend/server.py` (lines 266–278) and `frontend/algorithms.js` (lines 118–140).
  * **Reasoning Trace Rendering**: Frontend displays which rules fired, their conditions, and source guidelines under the "Knowledge Base Inference" card (`app.js` lines 216–230).

### D. Information Retrieval / Natural Language Processing (Phase 2)
* **Academic Criteria**: Document corpus indexing, query vectorization, and search using TF-IDF and cosine similarity.
* **Code Implementation**:
  * **Clinical Document Corpus**: Stored as a collection of 10 clinical guideline summaries. Defined in `frontend/algorithms.js` (lines 316–338).
  * **TF-IDF Vectorizer & Cosine Similarity**: Vectorizes search queries and compares dot products to retrieve the highest-scoring medical text. Implemented in `frontend/algorithms.js` (lines 340–366).
  * **Clinical Search Tab**: Rendered dynamically under the "Clinical KB" tab in `index.html` and managed by `app.js` (lines 559–604).

---

## 2. Mandatory Verification & Contrasting Cases
The project is verified against two contrasting patient profiles to prove adaptive behavior:

| Case Profile | Clinical Measurements | ML Risk Output | A* Care Pathway | KB Inference Output |
| :--- | :--- | :--- | :--- | :--- |
| **Case A: High Risk** | Age: 63, BP: 158, Chol: 275, Angina: Yes, ST Dep: 3.2 | **High Risk** (95.9% Prob) | 7 steps: High Risk $\rightarrow$ BP Management $\rightarrow$ Cholesterol Management $\rightarrow$ ECG $\rightarrow$ Cardiology Consult $\rightarrow$ Risk Strat $\rightarrow$ Goal | Fired 6 rules: Hypertension, hyperlipidemia, suspected CAD, critical screening. |
| **Case B: Low Risk** | Age: 38, BP: 112, Chol: 185, Angina: No, ST Dep: 0.1 | **Low Risk** (2.5% Prob) | 1 step: Low Risk $\rightarrow$ Goal State | Fired 2 rules: Routine monitoring, healthy lifestyle. |

*This contrasting execution is demonstrated programmatically in `notebooks/CardioAI_Phase2_Clean.ipynb` (Section 7) and can be checked interactively in the web application using the sidebar simulator profiles.*
