# CardioAI — Viva Preparation & Technical Q&A Guide

This guide is structured to help you prepare for viva examinations, project defenses, and technical evaluations. It includes a detailed folder breakdown and an extensive question-and-answer library covering Machine Learning, Artificial Intelligence, and Full-Stack Implementation.

---

## 1. Folder & File Breakdown Guide

| Directory / File Path | Purpose & Technical Role | Why It Exists |
| :--- | :--- | :--- |
| **`backend/`** | Backend API directory. | Isolates backend services from static assets. |
| `backend/server.py` | Python Flask application serving `/predict` and `/simulate`. | Acts as the system controller. Performs feature engineering, runs prediction via XGBoost, builds the medical graph, runs the python A* agent, executes forward-chaining logic, and formats the response. |
| `backend/models/` | Directory containing binary model weights. | Stores the serialized model outputs. |
| `backend/models/heart_disease_model.pkl` | Serialized XGBoost model weights. | Enables loading the pre-trained classifier for instant live predictions. |
| `backend/models/heart_scaler.pkl` | Serialized StandardScaler state. | Preserves mean and variance parameters to scale new inputs identically to training data. |
| **`frontend/`** | Frontend client directory. | Hosts the client-side user interface. |
| `frontend/index.html` | Core dashboard structure. | Builds the HTML grid layout, tabs, assessment form, and placeholder panels. |
| `frontend/styles.css` | Premium styling sheet. | Renders custom HSL colors, dark grids, glassmorphism, heartbeat animations, and responsive risk badges. |
| `frontend/app.js` | Frontend controller. | Binds event listeners to the DOM, handles AJAX fetch calls to Flask, manages the UI state, and routes to local scripts on connection failure. |
| `frontend/model.js` | Client-side ML model. | Approximates the XGBoost model locally and calculates feature sensitivity adjustments. |
| `frontend/algorithms.js` | Client-side AI algorithms. | Implements client-side A* Search, forward-chaining rules (R01-R24), and TF-IDF guideline query matches. |
| **`notebooks/`** | Research & prototyping directory. | Documents the data science and algorithm research steps. |
| `notebooks/phase_1.ipynb` | Supervised learning notebook. | Contains exploratory data analysis (EDA), SMOTE balancing, model evaluation, and training pipelines. |
| `notebooks/CardioAI_Phase2_Clean.ipynb` | Multi-Paradigm notebook. | Prototypes A* search, forward chaining rules, and TF-IDF search indexing in Python. |

---

## 2. Technical Q&A: Machine Learning (Phase 1)

### Q1: Why did you choose XGBoost over simpler classifiers like Logistic Regression?
* **Answer**: Tabular clinical datasets contain complex, non-linear interactions. For example, a high resting blood pressure might represent a higher risk for an elderly patient than a younger one, which is a non-linear relationship. While Logistic Regression assumes linear decision boundaries, XGBoost builds an ensemble of sequential decision trees that learn these non-linear feature splits, yielding a higher ROC-AUC ($0.898$).

### Q2: What is the purpose of the StandardScaler, and why is it saved as a pickle file?
* **Answer**: XGBoost is scale-invariant (since it uses tree splits), but your model pipeline was trained on features normalized via a StandardScaler to align other models (like Logistic Regression). When deploying, we must scale new patient values using the exact same mean and standard deviation calculated from the training data. Saving it as `heart_scaler.pkl` ensures we apply identical normalization in production.

### Q3: Why is the classification threshold set to 0.35 instead of 0.50?
* **Answer**: In medical screening, a False Negative (missing a patient with heart disease) can lead to serious consequences, whereas a False Positive (referring a healthy patient for secondary tests) is manageable. By reducing the threshold to `0.35` (calibrated via Youden's J statistic), we prioritize **Recall (Sensitivity)**, reaching `0.931` (93.1%), which ensures that high-risk cases are not missed.

### Q4: Explain the feature engineering you performed. Why are interaction terms useful?
* **Answer**: We engineered four features:
  1. `bp_chol_interaction`: Multiplying blood pressure and cholesterol captures metabolic risk.
  2. `exercise_stress_score`: Summing angina and ST depression represents cardiovascular strain during exercise.
  3. `thalch_age_ratio`: Relates maximum heart rate to the age-predicted physical maximum.
  4. `age_risk_group`: Groups age into clinical cohorts.
  Interaction terms combine related features into a single column, helping tree algorithms identify combined risks without requiring deep tree structures.

---

## 3. Technical Q&A: Artificial Intelligence Algorithms (Phase 2)

### Q5: How is A* Search applied to patient care pathways?
* **Answer**: We represent clinical routing as a pathfinding problem. The dynamic medical graph starts at the patient's ML-predicted risk state (`Low`, `Medium`, or `High Risk`) and ends at the `Goal State`. Intermediate nodes represent clinical interventions (ECG, Stress Test, Cardiology Consult). Graph edges represent transitions with step costs representing clinical resource consumption. A* finds the optimal sequence of steps to guide the patient from their risk state to a resolved treatment plan.

### Q6: What makes your heuristic function $h(n)$ admissible? Why is admissibility critical?
* **Answer**: A heuristic $h(n)$ is admissible if it never overestimates the actual cost to reach the goal state ($h(n) \le h^*(n)$). Our heuristic assign values that represent the physical minimum number of transition steps to the goal (e.g., `Risk Stratification` has $h(n) = 1$ because it takes at least $1$ step to reach `Goal State`). Admissibility guarantees that A* will find the mathematically optimal (lowest-cost) pathway.

### Q7: Explain the difference between Forward Chaining and Backward Chaining. Why use Forward Chaining?
* **Answer**: 
  * **Forward Chaining**: Starts with known facts and applies rules to infer new facts (data-driven).
  * **Backward Chaining**: Starts with a goal hypothesis and searches backward for supporting facts (goal-driven).
  Since the system starts with raw patient diagnostic values and aims to generate a complete report of inferred findings, Forward Chaining is the correct choice.

### Q8: How does the TF-IDF search engine query guidelines?
* **Answer**: Term Frequency-Inverse Document Frequency (TF-IDF) converts documents and user queries into numerical vectors based on word frequencies, down-weighting common words (like "the") while highlighting clinical terms (like "ischemia"). Cosine similarity measures the angle between the query vector and document vectors. The document vector closest to the query vector represents the most relevant guideline.

---

## 4. Technical Q&A: Full-Stack Implementation

### Q9: What is CORS, and why did we run a local HTTP server for the frontend?
* **Answer**: Cross-Origin Resource Sharing (CORS) is a browser security mechanism that blocks scripts on one origin (e.g., opening a local file via `file://`) from fetching resources from a different origin (e.g., the Flask backend on `http://localhost:5000`). By running a local HTTP server (`python -m http.server 8000`), the frontend runs on `http://localhost:8000`, enabling standard cross-origin requests.

### Q10: How does the offline fallback mode work?
* **Answer**: In `app.js`, fetch requests are wrapped in try-catch blocks. If a request to the Flask server fails (e.g., due to connection timeout), the catch block is executed. It switches the navbar status badge to `Offline (Edge AI)` and routes the input variables to the client-side JavaScript implementations of the ML model (`model.js`), A* search, and forward-chaining algorithms (`algorithms.js`), providing a seamless fallback.
